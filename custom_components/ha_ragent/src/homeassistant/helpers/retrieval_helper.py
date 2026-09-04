from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from custom_components.ha_ragent.src.models.retrieval.scored_result import ScoredResult
from custom_components.ha_ragent.src.models.retrieval.continuity_context import ContinuityContext
from custom_components.ha_ragent.src.models.retrieval.turn_context import TurnContext

T = TypeVar("T")


class RetrievalHelper:
    """Stateless helpers for building and reranking retrieval queries."""

    _ACTION_FAMILIES = {"power", "position", "lock", "light", "climate", "media"}
    _FAMILY_DOMAINS = {
        "position": {"cover"},
        "lock": {"lock"},
        "light": {"light"},
        "climate": {"climate"},
        "media": {"media_player"},
    }

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
    def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
        """Return cosine similarity for two embedding vectors."""
        left = list(left)
        right = list(right)
        if len(left) != len(right) or not left:
            return 0.0
        dot_product = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if not left_norm or not right_norm:
            return 0.0
        return dot_product / (left_norm * right_norm)

    @staticmethod
    def select_history_contexts(
        contexts: Iterable[TurnContext],
        vectors: dict[str, list[float]],
        current_vector: list[float],
        max_age_seconds: float = 300.0,
        limit: int = 3,
        now: float | None = None,
    ) -> list[tuple[TurnContext, float]]:
        """Select semantic history with decay plus a small short-term signal."""
        now = time.time() if now is None else now
        contexts = list(contexts)
        selected: dict[str, tuple[TurnContext, float]] = {}
        for index, context in enumerate(contexts):
            fallback_age = float((len(contexts) - index - 1) * 30)
            age = max(0.0, now - context.created_at) if context.created_at is not None else fallback_age
            if age > max_age_seconds:
                continue
            decay = 0.5 ** (age / 120.0)
            similarity = max(0.0, RetrievalHelper.cosine_similarity(
                current_vector,
                vectors.get(context.key, []),
            ))
            relevance = similarity * decay
            if context.has_canonical_context:
                relevance += 0.1 * decay
            if similarity >= 0.2:
                selected[context.key] = (context, relevance)

            # Keep canonical context from the last two very recent turns as a
            # separate short-term continuity signal, without parsing language.
            if context.has_canonical_context and index >= len(contexts) - 2 and age <= 90.0:
                short_term_weight = 0.15 * decay
                previous = selected.get(context.key)
                if previous is None or short_term_weight > previous[1]:
                    selected[context.key] = (context, short_term_weight)

        return sorted(selected.values(), key=lambda item: item[1], reverse=True)[:limit]

    @staticmethod
    def build_continuity_context(selected_contexts: Iterable[tuple[TurnContext, float]]) -> ContinuityContext:
        """Aggregate selected structured turns into weighted continuity maps."""
        continuity = ContinuityContext()
        for context, weight in selected_contexts:
            continuity.selected_turn_keys.add(context.key)
            for attribute in (
                "entities",
                "tools",
                "areas",
                "domains",
                "device_classes",
                "actions",
                "ambiguous_entities",
            ):
                values = getattr(context, attribute)
                target = getattr(continuity, attribute)
                for value in values:
                    normalized = str(value).casefold()
                    target[normalized] = max(target.get(normalized, 0.0), weight)
            continuity.target_groups.extend(
                (group, weight) for group in context.target_groups
            )
        return continuity

    @staticmethod
    def adaptive_candidate_limit(limit: int) -> int:
        """Return a bounded internal pool size for hybrid retrieval."""
        return min(40, max(limit * 4, limit + 8)) if limit > 0 else 0

    @staticmethod
    def expanded_tool_limit(limit: int) -> int:
        """Expand the exposed tool set for a confidently resolved target."""
        return min(12, limit * 2) if limit > 0 else 0

    @staticmethod
    def expanded_device_limit(limit: int, continuity: ContinuityContext) -> int:
        """Keep all recent successful entity targets within a bounded shortlist."""
        successful_entities = {
            entity.casefold()
            for group, _ in continuity.target_groups
            for entity in group.entities
        }
        return min(12, max(limit, len(successful_entities))) if limit > 0 else 0

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
    def trusted_location_score(device: Any, area: str = "", floor: str = "") -> float:
        """Prefer the requesting device's location as a bounded ranking signal."""
        device_area = str(getattr(device, "area_name", "") or "").casefold()
        device_floor = str(getattr(device, "floor_name", "") or "").casefold()
        score = 0.0
        if area and device_area == area.casefold():
            score += 1.0
        if floor and device_floor == floor.casefold():
            score += 0.5
        return score

    @staticmethod
    def reciprocal_rank_fusion(ranked_keys: Iterable[Iterable[str]], rank_constant: int = 60) -> dict[str, float]:
        """Fuse independent rankings using reciprocal rank fusion."""
        scores: dict[str, float] = {}
        for ranking in ranked_keys:
            for rank, key in enumerate(ranking, start=1):
                scores[key] = scores.get(key, 0.0) + 1.0 / (rank_constant + rank)
        return scores

    @staticmethod
    def _rank_positive_scores(scores: dict[str, float], minimum: float) -> list[str]:
        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        return [item_key for item_key, score in ranked if score >= minimum]

    @staticmethod
    def _has_strong_current_match(
        match_scores: dict[str, tuple[float, float]],
        vector_ranking: list[str],
        exact_ranking: list[str],
        fuzzy_ranking: list[str],
    ) -> bool:
        if any(
            exact >= 0.9 or fuzzy >= 0.9
            for exact, fuzzy in match_scores.values()
        ):
            return True
        if not vector_ranking:
            return False

        best_vector = vector_ranking[0]
        if exact_ranking and best_vector == exact_ranking[0]:
            return True
        return (
            bool(fuzzy_ranking)
            and best_vector == fuzzy_ranking[0]
            and match_scores[best_vector][1] >= 0.7
        )

    @staticmethod
    def _trim_to_confident_keys(
        ordered_keys: list[str],
        match_scores: dict[str, tuple[float, float]],
        vector_positions: dict[str, int],
        limit: int,
    ) -> list[str]:
        if not ordered_keys:
            return ordered_keys

        top_key = ordered_keys[0]
        top_exact, top_fuzzy = match_scores[top_key]
        top_vector_rank = vector_positions.get(top_key)
        top_is_confident = (
            top_exact >= 0.9
            or top_fuzzy >= 0.9
            or (
                top_vector_rank == 1
                and (top_exact >= 0.5 or top_fuzzy >= 0.7)
            )
        )
        if not top_is_confident:
            return ordered_keys

        confident = [
            item_key
            for item_key in ordered_keys[:limit]
            if match_scores[item_key][0] >= 0.5
            or match_scores[item_key][1] >= 0.7
            or vector_positions.get(item_key, limit + 1) <= 2
        ]
        return confident or ordered_keys

    @staticmethod
    def _merge_preserved_keys(selected_keys: list[str], preserved_keys: list[str], limit: int) -> list[str]:
        missing = [key for key in preserved_keys if key not in selected_keys]
        if not missing:
            return selected_keys
        retained = [
            key for key in selected_keys if key not in preserved_keys
        ][:max(0, limit - len(preserved_keys))]
        return [*retained, *preserved_keys]

    @staticmethod
    def rank_scored_candidates(
        vector_results: Iterable[ScoredResult[T]],
        lexical_items: Iterable[T],
        query: str,
        key: Callable[[T], str],
        text_parts: Callable[[T], Iterable[object]],
        limit: int,
        metadata_score: Callable[[T], float] | None = None,
        continuity_score: Callable[[T], float] | None = None,
        preserve_score: Callable[[T], float] | None = None,
    ) -> list[T]:
        """Fuse rank-based signals and suppress stale continuity on strong matches."""
        if limit <= 0:
            return []

        vector_results = list(vector_results)
        candidates: dict[str, T] = {
            key(result.item): result.item for result in vector_results
        }
        for item in lexical_items:
            candidates.setdefault(key(item), item)

        vector_ranking = [
            key(result.item)
            for result in sorted(vector_results, key=lambda result: result.rank)
        ]
        vector_positions = {
            item_key: position
            for position, item_key in enumerate(vector_ranking, start=1)
        }
        match_scores = {
            item_key: RetrievalHelper._match_scores(query, text_parts(item))
            for item_key, item in candidates.items()
        }
        exact_ranking = RetrievalHelper._rank_positive_scores(
            {item_key: scores[0] for item_key, scores in match_scores.items()},
            0.75,
        )
        fuzzy_ranking = RetrievalHelper._rank_positive_scores(
            {item_key: scores[1] for item_key, scores in match_scores.items()},
            0.2,
        )
        metadata_scores = {
            item_key: metadata_score(item)
            for item_key, item in candidates.items()
        } if metadata_score else {}
        metadata_ranking = RetrievalHelper._rank_positive_scores(
            {key: score for key, score in metadata_scores.items() if score > 0},
            0.0,
        )
        has_strong_current_match = RetrievalHelper._has_strong_current_match(
            match_scores,
            vector_ranking,
            exact_ranking,
            fuzzy_ranking,
        )

        continuity_scores = {
            item_key: continuity_score(item)
            for item_key, item in candidates.items()
        } if continuity_score and not has_strong_current_match else {}
        continuity_ranking = RetrievalHelper._rank_positive_scores(
            {key: score for key, score in continuity_scores.items() if score > 0},
            0.0,
        )

        fused = RetrievalHelper.reciprocal_rank_fusion(
            (vector_ranking, exact_ranking, fuzzy_ranking, metadata_ranking)
        )
        for rank, item_key in enumerate(continuity_ranking, start=1):
            fused[item_key] = fused.get(item_key, 0.0) + 0.25 / (60 + rank)
        for item_key, (exact_score, fuzzy_score) in match_scores.items():
            fused[item_key] = (
                fused.get(item_key, 0.0)
                + (0.01 * exact_score)
                + (0.01 * fuzzy_score)
            )
        ordered_keys = sorted(
            candidates,
            key=lambda item_key: (
                -fused.get(item_key, 0.0),
                vector_ranking.index(item_key)
                if item_key in vector_ranking
                else len(vector_ranking),
            ),
        )

        # Strong agreement permits a smaller, precise result. Weak confidence
        # retains the configured limit so downstream semantic search can recover.
        ordered_keys = RetrievalHelper._trim_to_confident_keys(
            ordered_keys,
            match_scores,
            vector_positions,
            limit,
        )

        selected_keys = ordered_keys[:limit]
        if preserve_score and not has_strong_current_match:
            preserved_keys = [
                item_key
                for item_key, score in sorted(
                    (
                        (item_key, preserve_score(item))
                        for item_key, item in candidates.items()
                    ),
                    key=lambda pair: pair[1],
                    reverse=True,
                )
                if score > 0
            ][:limit]
            selected_keys = RetrievalHelper._merge_preserved_keys(
                selected_keys,
                preserved_keys,
                limit,
            )

        return [candidates[item_key] for item_key in selected_keys]

    @staticmethod
    def target_is_confident(query: str, devices: Iterable[Any], continuity: ContinuityContext) -> bool:
        """Return whether the current or successful prior target is resolved."""
        devices = list(devices)
        for device in devices:
            exact, fuzzy = RetrievalHelper._match_scores(
                query,
                (
                    getattr(device, "id", ""),
                    getattr(device, "friendly_name", ""),
                    *(getattr(device, "aliases", None) or []),
                ),
            )
            if exact >= 0.9 or fuzzy >= 0.9:
                return True
        return any(
            continuity.successful_target_score(device) > 0
            for device in devices
        )

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
    def _device_value(device: Any, name: str, default: Any = None) -> Any:
        if isinstance(device, dict):
            return device.get(name, default)
        return getattr(device, name, default)

    @staticmethod
    def _device_domains(devices: Iterable[Any]) -> set[str]:
        domains: set[str] = set()
        for device in devices:
            values = RetrievalHelper._device_value(device, "domain", []) or []
            if isinstance(values, str):
                values = [values]
            domains.update(str(value).casefold() for value in values)
        return domains

    @staticmethod
    def _device_classes(devices: Iterable[Any]) -> set[str]:
        return {
            str(value).casefold()
            for device in devices
            if (value := RetrievalHelper._device_value(device, "device_class"))
        }

    @staticmethod
    def _metadata_value(tool: Any, name: str, default: Any = False) -> Any:
        metadata = getattr(tool, "metadata", None)
        if isinstance(metadata, dict):
            return metadata.get(name, default)
        return getattr(metadata, name, default)

    @staticmethod
    def _family_matches_domains(tool: Any, domains: set[str]) -> bool:
        family = str(getattr(tool, "family", "") or "").casefold()
        family_domains = RetrievalHelper._FAMILY_DOMAINS.get(family)
        return not family_domains or not domains or bool(domains & family_domains)

    @staticmethod
    def _is_action_tool(tool: Any) -> bool:
        family = str(getattr(tool, "family", "") or "").casefold()
        if family in RetrievalHelper._ACTION_FAMILIES:
            return True
        return any(
            RetrievalHelper._metadata_value(tool, name)
            for name in ("is_domain_aware", "is_area_aware", "is_device_class_aware")
        )

    @staticmethod
    def tool_device_compatibility(tool: Any, devices: Iterable[Any]) -> float:
        """Score whether a tool schema can target the retrieved devices."""
        devices = list(devices)
        if not devices:
            return 0.0
        properties = (tool.parameters or {}).get("properties") or {}
        domains = RetrievalHelper._device_domains(devices)
        device_classes = RetrievalHelper._device_classes(devices)
        score = 0.0
        allowed_domains = RetrievalHelper._schema_values(properties.get("domain", {}))
        allowed_classes = RetrievalHelper._schema_values(properties.get("device_class", {}))
        if allowed_domains:
            score += 2.0 if domains & allowed_domains else -2.0
        if allowed_classes:
            score += 2.0 if device_classes & allowed_classes else -2.0

        if RetrievalHelper._metadata_value(tool, "is_domain_aware"):
            score += 0.5
        if RetrievalHelper._metadata_value(tool, "is_device_class_aware") and device_classes:
            score += 0.5
        has_area = any(
            RetrievalHelper._device_value(device, "area_name")
            or RetrievalHelper._device_value(device, "area")
            for device in devices
        )
        if RetrievalHelper._metadata_value(tool, "is_area_aware") and has_area:
            score += 0.25
        return score

    @staticmethod
    def tool_is_compatible(tool: Any, devices: Iterable[Any]) -> bool:
        """Reject schema or action-family constraints matching no device."""
        devices = list(devices)
        if not devices:
            return True
        properties = (tool.parameters or {}).get("properties") or {}
        domains = RetrievalHelper._device_domains(devices)
        device_classes = RetrievalHelper._device_classes(devices)
        allowed_domains = RetrievalHelper._schema_values(properties.get("domain", {}))
        allowed_classes = RetrievalHelper._schema_values(properties.get("device_class", {}))
        if allowed_domains and not domains.intersection(allowed_domains):
            return False
        if allowed_classes and not device_classes.intersection(allowed_classes):
            return False
        return RetrievalHelper._family_matches_domains(tool, domains)

    @staticmethod
    def has_compatible_action_tool(tools: Iterable[Any], devices: Iterable[Any]) -> bool:
        """Return whether the shortlist contains an action for the target devices."""
        devices = list(devices)
        for tool in tools:
            if not RetrievalHelper.tool_is_compatible(tool, devices):
                continue
            if RetrievalHelper._is_action_tool(tool):
                return True
        return False

    @staticmethod
    def rerank_tools_for_devices(tools: Iterable[T], devices: Iterable[Any], limit: int) -> list[T]:
        """Hard-filter incompatible tools, then fuse compatible tool ranks."""
        devices = list(devices)
        tools = [
            tool
            for tool in tools
            if RetrievalHelper.tool_is_compatible(tool, devices)
        ]
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
