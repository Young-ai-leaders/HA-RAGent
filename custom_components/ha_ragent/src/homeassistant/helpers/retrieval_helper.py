from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from custom_components.ha_ragent.src.models.scored_result import ScoredResult

T = TypeVar("T")


class RetrievalHelper:
    """Stateless helpers for building and reranking retrieval queries."""

    @staticmethod
    def _normalize(text: object) -> str:
        value = str(text or "").casefold()
        return " ".join("".join(
            character if character.isalnum() else " "
            for character in value
        ).split())

    @staticmethod
    def build_retrieval_text(current_request: str) -> str:
        """Build a language-neutral query from only the current request."""
        return " ".join(current_request.split())

    @staticmethod
    def adaptive_candidate_limit(limit: int) -> int:
        """Return a bounded internal pool size for hybrid retrieval."""
        return min(40, max(limit * 4, limit + 8)) if limit > 0 else 0

    @staticmethod
    def _character_ngrams(text: str, size: int = 3) -> set[str]:
        compact = text.replace(" ", "")
        if len(compact) <= size:
            return {compact} if compact else set()
        return {compact[index:index + size] for index in range(len(compact) - size + 1)}

    @staticmethod
    def _match_scores(query: str, values: Iterable[object]) -> tuple[float, float]:
        query_text = RetrievalHelper._normalize(query)
        query_tokens = set(query_text.split())
        query_ngrams = RetrievalHelper._character_ngrams(query_text)
        exact_score = 0.0
        fuzzy_score = 0.0
        for value in values:
            normalized = RetrievalHelper._normalize(value)
            if not normalized:
                continue
            value_tokens = set(normalized.split())
            if normalized == query_text:
                exact_score = max(exact_score, 1.0)
            elif normalized in query_text:
                exact_score = max(exact_score, 0.9)
            elif value_tokens:
                exact_score = max(
                    exact_score,
                    len(query_tokens & value_tokens) / len(value_tokens),
                )

            value_ngrams = RetrievalHelper._character_ngrams(normalized)
            denominator = len(query_ngrams) + len(value_ngrams)
            if denominator:
                fuzzy_score = max(
                    fuzzy_score,
                    2.0 * len(query_ngrams & value_ngrams) / denominator,
                )
        return exact_score, fuzzy_score

    @staticmethod
    def field_match_score(query: str, values: Iterable[object]) -> float:
        """Score structured metadata using language-neutral matching."""
        exact, fuzzy = RetrievalHelper._match_scores(query, values)
        return exact + (0.5 * fuzzy)

    @staticmethod
    def reciprocal_rank_fusion(
        ranked_keys: Iterable[Iterable[str]],
        rank_constant: int = 60,
    ) -> dict[str, float]:
        """Fuse independent rankings using reciprocal rank fusion."""
        scores: dict[str, float] = {}
        for ranking in ranked_keys:
            for rank, key in enumerate(ranking, start=1):
                scores[key] = scores.get(key, 0.0) + 1.0 / (rank_constant + rank)
        return scores

    @staticmethod
    def rank_scored_candidates(
        vector_results: Iterable[ScoredResult[T]],
        lexical_items: Iterable[T],
        query: str,
        key: Callable[[T], str],
        text_parts: Callable[[T], Iterable[object]],
        limit: int,
        metadata_score: Callable[[T], float] | None = None,
    ) -> list[T]:
        """Fuse vector, exact, fuzzy, and metadata rankings with adaptive top-k."""
        if limit <= 0:
            return []

        vector_results = list(vector_results)
        candidates: dict[str, T] = {key(result.item): result.item for result in vector_results}
        for item in lexical_items:
            candidates.setdefault(key(item), item)

        vector_scores = {key(result.item): result.score for result in vector_results}
        vector_ranking = [key(result.item) for result in sorted(vector_results, key=lambda result: result.rank)]
        match_scores = {
            item_key: RetrievalHelper._match_scores(query, text_parts(item))
            for item_key, item in candidates.items()
        }
        exact_ranking = [
            item_key for item_key, score in sorted(
                ((item_key, scores[0]) for item_key, scores in match_scores.items() if scores[0] >= 0.75),
                key=lambda pair: pair[1],
                reverse=True,
            )
        ]
        fuzzy_ranking = [
            item_key for item_key, score in sorted(
                ((item_key, scores[1]) for item_key, scores in match_scores.items() if scores[1] >= 0.2),
                key=lambda pair: pair[1],
                reverse=True,
            )
        ]
        metadata_scores = {
            item_key: metadata_score(item)
            for item_key, item in candidates.items()
        } if metadata_score else {}
        metadata_ranking = [
            item_key for item_key, score in sorted(
                ((item_key, score) for item_key, score in metadata_scores.items() if score > 0),
                key=lambda pair: pair[1],
                reverse=True,
            )
        ]

        fused = RetrievalHelper.reciprocal_rank_fusion(
            (vector_ranking, exact_ranking, fuzzy_ranking, metadata_ranking)
        )
        for item_key, vector_score in vector_scores.items():
            fused[item_key] = fused.get(item_key, 0.0) + (0.01 * vector_score)
        for item_key, (exact_score, fuzzy_score) in match_scores.items():
            fused[item_key] = fused.get(item_key, 0.0) + (0.01 * exact_score) + (0.01 * fuzzy_score)
        ordered_keys = sorted(
            candidates,
            key=lambda item_key: (-fused.get(item_key, 0.0), vector_ranking.index(item_key) if item_key in vector_ranking else len(vector_ranking)),
        )

        # Strong agreement permits a smaller, precise result. Weak confidence
        # retains the configured limit so downstream semantic search can recover.
        top_key = ordered_keys[0] if ordered_keys else None
        if top_key:
            top_exact, top_fuzzy = match_scores[top_key]
            top_vector = vector_scores.get(top_key, 0.0)
            if top_exact >= 0.9 or (top_fuzzy >= 0.85 and top_vector >= 0.55):
                confident = [
                    item_key for item_key in ordered_keys[:limit]
                    if match_scores[item_key][0] >= 0.5
                    or match_scores[item_key][1] >= 0.7
                    or vector_scores.get(item_key, 0.0) >= max(0.45, top_vector - 0.12)
                ]
                if confident:
                    ordered_keys = confident

        return [candidates[item_key] for item_key in ordered_keys[:limit]]

    @staticmethod
    def _schema_values(schema: object) -> set[str]:
        values: set[str] = set()
        if isinstance(schema, dict):
            for name, value in schema.items():
                if name == "const" and isinstance(value, (str, int, float)):
                    values.add(str(value).casefold())
                elif name == "enum" and isinstance(value, list):
                    values.update(str(item).casefold() for item in value)
                else:
                    values.update(RetrievalHelper._schema_values(value))
        elif isinstance(schema, list):
            for value in schema:
                values.update(RetrievalHelper._schema_values(value))
        return values

    @staticmethod
    def tool_device_compatibility(tool: Any, devices: Iterable[Any]) -> float:
        """Score whether a tool schema can target the retrieved devices."""
        devices = list(devices)
        if not devices:
            return 0.0
        properties = (tool.parameters or {}).get("properties") or {}
        def device_value(device: Any, name: str, default: Any = None) -> Any:
            return device.get(name, default) if isinstance(device, dict) else getattr(device, name, default)

        domains = {
            str(domain).casefold()
            for device in devices
            for domain in (device_value(device, "domain", []) or [])
        }
        device_classes = {
            str(device_value(device, "device_class")).casefold()
            for device in devices
            if device_value(device, "device_class")
        }
        score = 0.0
        allowed_domains = RetrievalHelper._schema_values(properties.get("domain", {}))
        allowed_classes = RetrievalHelper._schema_values(properties.get("device_class", {}))
        if allowed_domains:
            score += 2.0 if domains & allowed_domains else -2.0
        if allowed_classes:
            score += 2.0 if device_classes & allowed_classes else -2.0

        metadata = tool.metadata
        get_metadata = (
            metadata.get
            if isinstance(metadata, dict)
            else lambda name, default=False: getattr(metadata, name, default)
        )
        if get_metadata("is_domain_aware"):
            score += 0.5
        if get_metadata("is_device_class_aware") and device_classes:
            score += 0.5
        if get_metadata("is_area_aware") and any(device_value(device, "area_name") or device_value(device, "area") for device in devices):
            score += 0.25
        return score

    @staticmethod
    def rerank_tools_for_devices(tools: Iterable[T], devices: Iterable[Any], limit: int) -> list[T]:
        """Fuse existing tool rank with device compatibility rank."""
        tools = list(tools)
        devices = list(devices)
        original = [str(index) for index in range(len(tools))]
        compatible = sorted(
            (
                (index, tool)
                for index, tool in enumerate(tools)
                if RetrievalHelper.tool_device_compatibility(tool, devices) > 0
            ),
            key=lambda pair: RetrievalHelper.tool_device_compatibility(pair[1], devices),
            reverse=True,
        )
        compatibility = [str(index) for index, _ in compatible]
        fused = RetrievalHelper.reciprocal_rank_fusion((original, compatibility))
        return [
            tool for index, tool in sorted(
                enumerate(tools),
                key=lambda pair: (-fused[str(pair[0])], pair[0]),
            )[:limit]
        ]
