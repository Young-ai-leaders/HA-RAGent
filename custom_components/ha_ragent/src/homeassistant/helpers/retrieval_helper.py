from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from custom_components.ha_ragent.src.models.retrieval.scored_result import ScoredResult
from custom_components.ha_ragent.src.models.retrieval.continuity_context import ContinuityContext
from custom_components.ha_ragent.src.models.retrieval.turn_context import TurnContext
from custom_components.ha_ragent.src.models.embedding.tool_metadata import (
    CANONICAL_ACTION_ALIASES,
    canonical_actions_from_text,
    canonical_action_from_text,
    normalize_canonical_text,
    split_canonical_name,
)

T = TypeVar("T")


class RetrievalHelper:
    """Stateless helpers for building and reranking retrieval queries."""

    _FAMILY_DOMAINS = {
        "position": {"cover"},
        "lock": {"lock"},
        "light": {"light"},
        "climate": {"climate"},
        "media": {"media_player"},
    }
    _DOMAIN_ALIASES = {
        "light": {"light", "lights", "lamp", "lamps"},
        "switch": {"switch", "switches", "plug", "plugs"},
        "fan": {"fan", "fans"},
        "cover": {"cover", "covers", "blind", "blinds", "shade", "shades"},
        "lock": {"lock", "locks", "door", "doors"},
        "climate": {"climate", "thermostat", "thermostats", "heating"},
        "media_player": {"media", "player", "players", "speaker", "speakers"},
        "timer": {"timer", "timers"},
    }
    _SEARCH_STOP_WORDS = {
        "a",
        "an",
        "all",
        "device",
        "devices",
        "it",
        "please",
        "the",
        "them",
    }
    _FOLLOWUP_REFERENCES = ("it", "them", "same room", "there", "again", "the ones")
    _LOCATION_FOLLOWUP_PREFIXES = ("at ", "at the ", "in ", "in the ")
    _TOOL_SIGNAL_WEIGHTS = {
        "semantic_rank": 0.75,
        "semantic_similarity": 1.0,
        "lexical_exact": 2.0,
        "lexical_fuzzy": 0.75,
        "action_intent": 3.0,
        "domain": 1.5,
        "device_metadata": 0.5,
        "continuity": 0.5,
    }

    @staticmethod
    def _normalize(text: object) -> str:
        return normalize_canonical_text(text)

    @staticmethod
    def requested_action(query: str) -> str:
        """Return the explicit canonical action requested by the user."""
        return canonical_action_from_text(query)

    @staticmethod
    def requested_actions(query: str) -> tuple[str, ...]:
        """Return actions explicitly requested in the original request."""
        return canonical_actions_from_text(query)

    @staticmethod
    def _remove_action_phrases(query: str, action: str) -> str:
        """Remove action aliases before extracting target metadata."""
        if not action:
            return query
        padded = f" {query} "
        for alias in CANONICAL_ACTION_ALIASES[action]:
            marker = f" {alias} "
            padded = padded.replace(marker, " ")
        return padded.strip()

    @staticmethod
    def requested_domains(query: str) -> set[str]:
        """Return canonical Home Assistant domains explicitly named in a query."""
        normalized = RetrievalHelper._normalize(query)
        action = RetrievalHelper.requested_action(normalized)
        target_text = RetrievalHelper._remove_action_phrases(normalized, action)
        tokens = set(target_text.split())
        return {
            domain
            for domain, aliases in RetrievalHelper._DOMAIN_ALIASES.items()
            if tokens & aliases
        }

    @staticmethod
    def canonical_search_signature(query: object) -> str:
        """Collapse near-equivalent action searches into a stable cache key."""
        normalized = RetrievalHelper._normalize(query)
        action = RetrievalHelper.requested_action(normalized)
        domains = RetrievalHelper.requested_domains(normalized)
        explicit_action = any(
            f" {alias} " in f" {normalized} "
            for alias in CANONICAL_ACTION_ALIASES.get(action, ())
            if " " in alias
        )
        weak_power_search = action in {"on", "off", "toggle"} and not explicit_action
        signature_action = "power" if weak_power_search else action
        if weak_power_search:
            domains = set()
        ignored = set(RetrievalHelper._SEARCH_STOP_WORDS)
        ignored.update(alias for aliases in RetrievalHelper._DOMAIN_ALIASES.values() for alias in aliases)
        if weak_power_search:
            ignored.update(
                word
                for power_action in ("on", "off", "toggle")
                for alias in CANONICAL_ACTION_ALIASES[power_action]
                for word in alias.split()
            )
        elif action:
            ignored.update(
                word
                for alias in CANONICAL_ACTION_ALIASES[action]
                for word in alias.split()
            )
        remaining = sorted(token for token in normalized.split() if token not in ignored)
        return "|".join((signature_action, ",".join(sorted(domains)), " ".join(remaining)))

    @staticmethod
    def build_tool_search_query(
        trusted_query: str,
        fallback_query: str,
        devices: Iterable[Any],
    ) -> str:
        """Build a compact capability query from action and resolved device domains."""
        action = RetrievalHelper.requested_action(trusted_query)
        if not action:
            action = RetrievalHelper.requested_action(fallback_query)
        domains = RetrievalHelper._device_domains(devices)
        if not domains:
            domains = RetrievalHelper.requested_domains(trusted_query)
        if not action:
            return trusted_query or fallback_query

        aliases = CANONICAL_ACTION_ALIASES.get(action, ())
        sections = [f"canonical action: {action}"]
        if aliases:
            sections.append(f"action aliases: {' | '.join(aliases)}")
        if domains:
            sections.append(f"supported domains: {' | '.join(sorted(domains))}")
        return " | ".join(sections)

    @staticmethod
    def _followup_target_terms(group: Any) -> tuple[str, ...]:
        """Derive reusable target words without carrying the prior location."""
        ignored = {
            RetrievalHelper._normalize(value)
            for value in (*group.areas, *group.floors, *group.domains)
            if value
        }
        terms: list[str] = []
        for entity in group.entities:
            entity_name = str(entity).split(".", 1)[-1]
            terms.extend(
                part
                for part in RetrievalHelper._normalize(entity_name).split()
                if part not in ignored
            )
        return tuple(dict.fromkeys(terms))

    @staticmethod
    def resolve_followup_query(query: str, continuity: ContinuityContext) -> str:
        """Add compact successful target context for explicit follow-up references."""
        normalized = f" {RetrievalHelper._normalize(query)} "
        normalized_query = normalized.strip()
        is_reference = any(
            f" {phrase} " in normalized
            for phrase in RetrievalHelper._FOLLOWUP_REFERENCES
        )
        is_location_followup = (
            not RetrievalHelper.requested_action(query)
            and any(
                normalized_query.startswith(prefix)
                for prefix in RetrievalHelper._LOCATION_FOLLOWUP_PREFIXES
            )
        )
        if not is_reference and not is_location_followup:
            return query
        if not continuity.target_groups:
            return query

        group, _ = continuity.target_groups[0]
        if is_location_followup:
            context_parts = [
                *(f"target={value}" for value in RetrievalHelper._followup_target_terms(group)),
                *(f"domain={value}" for value in group.domains),
                *(f"device_class={value}" for value in group.device_classes),
            ]
        else:
            context_parts = [
                *(f"entity={value}" for value in group.entities),
                *(f"area={value}" for value in group.areas),
                *(f"floor={value}" for value in group.floors),
                *(f"domain={value}" for value in group.domains),
                *(f"device_class={value}" for value in group.device_classes),
            ]
        inherit_action = (
            (" again " in normalized or is_location_followup)
            and not RetrievalHelper.requested_action(query)
        )
        if group.action and inherit_action:
            context_parts.append(f"previous_action={group.action}")
        if not context_parts:
            return query
        return f"{query}\nRecent successful context: {' | '.join(context_parts)}"

    @staticmethod
    def _candidate_identity_values(device: Any) -> tuple[object, ...]:
        aliases = RetrievalHelper._device_value(device, "aliases", []) or []
        if isinstance(aliases, str):
            aliases = [aliases]
        return (
            RetrievalHelper._device_value(device, "id", ""),
            RetrievalHelper._device_value(device, "name", ""),
            RetrievalHelper._device_value(device, "friendly_name", ""),
            *aliases,
        )

    @staticmethod
    def _candidate_location_values(device: Any) -> tuple[object, ...]:
        return (
            RetrievalHelper._device_value(device, "area_name", ""),
            RetrievalHelper._device_value(device, "area", ""),
            RetrievalHelper._device_value(device, "floor_name", ""),
            RetrievalHelper._device_value(device, "floor", ""),
        )

    @staticmethod
    def _query_requests_group(query: str, domains: set[str]) -> bool:
        normalized = RetrievalHelper._normalize(query)
        tokens = set(normalized.split())
        if tokens & {"all", "both", "every"}:
            return True
        return any(
            plural in tokens
            for domain in domains
            for plural in RetrievalHelper._DOMAIN_ALIASES.get(domain, set())
            if plural.endswith("s")
        )

    @staticmethod
    def device_resolution(query: str, devices: Iterable[Any]) -> tuple[str, tuple[str, ...]]:
        """Resolve devices independently from retrieval rank for execution safety."""
        devices = list(devices)
        if not devices:
            return "weak", ()
        normalized_query = RetrievalHelper._normalize(query)
        requested_domains = RetrievalHelper.requested_domains(query)
        location_scope = (
            normalized_query.rsplit(" in ", 1)[-1]
            if " in " in normalized_query
            else normalized_query
        )
        requested_locations = {
            RetrievalHelper._normalize(value)
            for device in devices
            for value in RetrievalHelper._candidate_location_values(device)
            if value and RetrievalHelper._normalize(value) in location_scope
        }
        location_tokens = set(location_scope.split())
        temporal_location = any(token.isdigit() for token in location_tokens) or bool(
            location_tokens & {"hour", "hours", "minute", "minutes", "second", "seconds"}
        )
        location_required = bool(requested_locations) or (
            " in " in normalized_query
            and not temporal_location
        )
        scored: list[tuple[float, float, float, float, str]] = []
        for device in devices:
            identity, identity_fuzzy = RetrievalHelper._match_scores(
                query,
                RetrievalHelper._candidate_identity_values(device),
            )
            domains = RetrievalHelper._device_domains((device,))
            domain_match = 1.0 if requested_domains and domains & requested_domains else 0.0
            location_values = {
                RetrievalHelper._normalize(value)
                for value in RetrievalHelper._candidate_location_values(device)
                if value
            }
            location_match = 1.0 if requested_locations & location_values else 0.0
            name = str(
                RetrievalHelper._device_value(device, "id", "")
                or RetrievalHelper._device_value(device, "name", "")
            )
            score = 3.0 * max(identity, identity_fuzzy) + 1.5 * domain_match + 2.0 * location_match
            scored.append((score, max(identity, identity_fuzzy), domain_match, location_match, name))
        scored.sort(reverse=True)

        exact = [
            item
            for item in scored
            if item[1] >= 0.9
            and (not requested_domains or item[2] >= 1.0)
            and (not location_required or item[3] >= 1.0)
        ]
        if len(exact) == 1:
            return "high", (exact[0][4],)

        compatible = [
            item
            for item in scored
            if (not requested_domains or item[2] >= 1.0)
            and (not location_required or item[3] >= 1.0)
        ]
        if compatible and RetrievalHelper._query_requests_group(query, requested_domains):
            names = tuple(item[4] for item in compatible if item[4])
            return "group", names
        if len(compatible) == 1:
            return "high", (compatible[0][4],)
        if len(compatible) > 1:
            names = tuple(item[4] for item in compatible if item[4])
            return "ambiguous", names
        if len(scored) == 1 and scored[0][0] > 0 and not requested_domains and not location_required:
            return "high", (scored[0][4],)
        return "weak", tuple(item[4] for item in scored if item[4])

    @staticmethod
    def reduce_confident_devices(query: str, devices: Iterable[T]) -> list[T]:
        """Expose only independently resolved devices when confidence is high."""
        devices = list(devices)
        status, names = RetrievalHelper.device_resolution(query, devices)
        if status not in {"high", "group"}:
            return devices
        selected = set(names)
        return [
            device
            for device in devices
            if str(
                RetrievalHelper._device_value(device, "id", "")
                or RetrievalHelper._device_value(device, "name", "")
            ) in selected
        ]

    @staticmethod
    def select_device_candidates(query: str, devices: Iterable[T], limit: int) -> list[T]:
        """Apply ambiguity-aware exposure after broad device retrieval."""
        devices = list(devices)
        status, _ = RetrievalHelper.device_resolution(query, devices)
        if status == "high":
            return RetrievalHelper.reduce_confident_devices(query, devices)[:1]
        if status == "group":
            return RetrievalHelper.reduce_confident_devices(query, devices)[:max(2, limit)]
        if status == "ambiguous":
            return devices[:max(2, limit)]
        return devices[:limit]

    @staticmethod
    def build_retrieval_text(current_request: str) -> str:
        """Build a language-neutral query from only the current request."""
        return " ".join(current_request.split())

    @staticmethod
    def is_clarification(query: str, pending: str = "") -> bool:
        """Return whether a short turn refines an unresolved request."""
        normalized = RetrievalHelper._normalize(query)
        query_tokens = set(normalized.split()) - RetrievalHelper._SEARCH_STOP_WORDS
        pending_tokens = set(RetrievalHelper._normalize(pending).split()) - RetrievalHelper._SEARCH_STOP_WORDS
        return (
            not RetrievalHelper.requested_action(query)
            and (
                any(normalized.startswith(prefix) for prefix in RetrievalHelper._LOCATION_FOLLOWUP_PREFIXES)
                or any(f" {phrase} " in f" {normalized} " for phrase in RetrievalHelper._FOLLOWUP_REFERENCES)
                or bool(query_tokens & pending_tokens)
                or len(query_tokens) == 1
            )
        )

    @staticmethod
    def merge_pending_request(pending: str, clarification: str) -> str:
        """Merge a clarification without changing the pending intent."""
        return f"{pending}\nUser clarification: {clarification}" if pending else clarification

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
        selected_contexts = list(selected_contexts)
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
        recent_contexts = sorted(
            selected_contexts,
            key=lambda item: item[0].created_at or 0.0,
            reverse=True,
        )
        continuity.target_groups = [
            (group, weight)
            for context, weight in recent_contexts
            for group in context.target_groups
        ]
        return continuity

    @staticmethod
    def adaptive_candidate_limit(limit: int) -> int:
        """Return a bounded internal pool size for hybrid retrieval."""
        return min(64, max(limit * 6, limit + 12)) if limit > 0 else 0

    @staticmethod
    def expanded_tool_limit(limit: int) -> int:
        """Expand the exposed tool set for a confidently resolved target."""
        return min(20, limit * 3) if limit > 0 else 0

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
    def device_target_score(query: str, device: Any) -> float:
        """Score target identity and capability above location-only similarity."""
        aliases = RetrievalHelper._device_value(device, "aliases", []) or []
        domains = RetrievalHelper._device_value(device, "domain", []) or []
        if isinstance(aliases, str):
            aliases = [aliases]
        if isinstance(domains, str):
            domains = [domains]
        return RetrievalHelper.field_match_score(query, (
            RetrievalHelper._device_value(device, "id", ""),
            RetrievalHelper._device_value(device, "friendly_name", ""),
            *aliases,
            *domains,
            RetrievalHelper._device_value(device, "device_class", ""),
        ))

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
        trim_confident: bool = True,
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
        if trim_confident:
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
    def _tool_action(tool: Any) -> str:
        action = str(getattr(tool, "canonical_action", "") or "")
        if action:
            return action
        searchable = " ".join(
            (
                str(getattr(tool, "name", "") or ""),
                str(getattr(tool, "description", "") or ""),
            )
        )
        return canonical_action_from_text(searchable)

    @staticmethod
    def _tool_declared_domains(tool: Any) -> set[str]:
        properties = (getattr(tool, "parameters", None) or {}).get("properties") or {}
        domains = RetrievalHelper._schema_values(properties.get("domain", {}))
        name_tokens = set(split_canonical_name(getattr(tool, "name", "")))
        domains.update(
            domain
            for domain, aliases in RetrievalHelper._DOMAIN_ALIASES.items()
            if name_tokens & aliases
        )
        return domains

    @staticmethod
    def _tool_action_signal(tool: Any, requested_action: str, query: str) -> float:
        tool_action = RetrievalHelper._tool_action(tool)
        if not requested_action:
            if not tool_action:
                return 0.0
            exact, fuzzy = RetrievalHelper._match_scores(
                query,
                CANONICAL_ACTION_ALIASES.get(tool_action, ()),
            )
            if exact < 0.25 and fuzzy < 0.35:
                return 0.0
            return min(0.8, exact + (0.5 * fuzzy))
        if tool_action == requested_action:
            return 1.0
        if not tool_action:
            return 0.15
        if tool_action in {"on", "off", "toggle"} and requested_action in {"on", "off", "toggle"}:
            return -0.35
        return -0.2

    @staticmethod
    def _tool_domain_signal(tool: Any, requested_domains: set[str]) -> float:
        if not requested_domains:
            return 0.0
        declared_domains = RetrievalHelper._tool_declared_domains(tool)
        if declared_domains:
            return 1.0 if declared_domains & requested_domains else -0.35
        family = str(getattr(tool, "family", "") or "").casefold()
        family_domains = RetrievalHelper._FAMILY_DOMAINS.get(family)
        if family_domains:
            return 0.8 if family_domains & requested_domains else -0.3
        return 0.35

    @staticmethod
    def tool_ranking_signals(
        tool: Any,
        query: str,
        devices: Iterable[Any] = (),
        semantic_rank: int | None = None,
        semantic_score: float | None = None,
        continuity: float = 0.0,
    ) -> dict[str, float]:
        """Return independent, extensible signals used to rank a tool."""
        devices = list(devices)
        requested_domains = RetrievalHelper._device_domains(devices)
        if not requested_domains:
            requested_domains = RetrievalHelper.requested_domains(query)
        exact, fuzzy = RetrievalHelper._match_scores(
            query,
            getattr(tool, "canonical_search_parts", ()) or (
                getattr(tool, "name", ""),
                getattr(tool, "description", ""),
            ),
        )
        compatibility = RetrievalHelper.tool_device_compatibility(tool, devices)
        return {
            "semantic_rank": 1.0 / semantic_rank if semantic_rank else 0.0,
            "semantic_similarity": max(0.0, min(1.0, semantic_score or 0.0)),
            "lexical_exact": exact,
            "lexical_fuzzy": fuzzy,
            "action_intent": RetrievalHelper._tool_action_signal(
                tool,
                RetrievalHelper.requested_action(query),
                query,
            ),
            "domain": RetrievalHelper._tool_domain_signal(tool, requested_domains),
            "device_metadata": max(-1.0, min(1.0, compatibility / 2.0)),
            "continuity": max(0.0, continuity),
        }

    @staticmethod
    def tool_signal_score(signals: dict[str, float]) -> float:
        """Combine named tool-ranking signals using centralized weights."""
        return sum(
            RetrievalHelper._TOOL_SIGNAL_WEIGHTS.get(name, 0.0) * value
            for name, value in signals.items()
        )

    @staticmethod
    def rank_tool_candidates(
        vector_results: Iterable[ScoredResult[T]],
        lexical_tools: Iterable[T],
        query: str,
        devices: Iterable[Any],
        limit: int,
        continuity_score: Callable[[T], float] | None = None,
    ) -> list[T]:
        """Rank a broad tool pool without discarding uncertain candidates."""
        if limit <= 0:
            return []
        vector_results = list(vector_results)
        candidate_by_name = {
            str(getattr(tool, "name", "")): tool
            for tool in lexical_tools
        }
        semantic_ranks: dict[str, int] = {}
        semantic_scores: dict[str, float] = {}
        for result in vector_results:
            name = str(getattr(result.item, "name", ""))
            candidate_by_name.setdefault(name, result.item)
            semantic_ranks[name] = result.rank
            semantic_scores[name] = result.score

        scored: list[tuple[float, int, str, T]] = []
        for name, tool in candidate_by_name.items():
            continuity = continuity_score(tool) if continuity_score else 0.0
            signals = RetrievalHelper.tool_ranking_signals(
                tool,
                query,
                devices,
                semantic_rank=semantic_ranks.get(name),
                semantic_score=semantic_scores.get(name),
                continuity=continuity,
            )
            scored.append(
                (
                    RetrievalHelper.tool_signal_score(signals),
                    semantic_ranks.get(name, len(semantic_ranks) + 1),
                    name,
                    tool,
                )
            )
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        if len(scored) > 1:
            top_signals = RetrievalHelper.tool_ranking_signals(scored[0][3], query, devices)
            if top_signals["action_intent"] >= 1.0 and scored[0][0] - scored[1][0] >= 1.5:
                return [scored[0][3]]
        return [tool for _, _, _, tool in scored[:limit]]

    @staticmethod
    def tool_search_confidence(tools: Iterable[Any], query: str, devices: Iterable[Any]) -> str:
        """Classify the top result without turning uncertainty into a hard failure."""
        tools = list(tools)
        if not tools:
            return "none"
        top_signals = RetrievalHelper.tool_ranking_signals(tools[0], query, devices)
        if top_signals["action_intent"] >= 1.0 and top_signals["domain"] >= 0.35:
            return "high"
        if (
            top_signals["action_intent"] >= 1.0
            or top_signals["domain"] >= 0.8
            or top_signals["lexical_exact"] >= 0.5
        ):
            return "medium"
        return "low"

    @staticmethod
    def rank_tools_for_query(tools: Iterable[T], query: str, devices: Iterable[Any] = ()) -> list[T]:
        """Order tools by soft intent signals while retaining uncertain candidates."""
        devices = list(devices)
        return sorted(
            tools,
            key=lambda tool: RetrievalHelper.tool_signal_score(
                RetrievalHelper.tool_ranking_signals(tool, query, devices)
            ),
            reverse=True,
        )

    @staticmethod
    def build_tool_candidate_pool(
        vector_results: Iterable[ScoredResult[T]],
        lexical_tools: Iterable[T],
        query: str,
        devices: Iterable[Any] = (),
    ) -> tuple[list[ScoredResult[T]], list[T]]:
        """Build a complete tool pool while preserving vector-rank metadata."""
        vector_results = list(vector_results)
        candidate_by_name = {
            str(getattr(tool, "name", "")): tool
            for tool in lexical_tools
        }
        for result in vector_results:
            candidate_by_name.setdefault(str(getattr(result.item, "name", "")), result.item)
        candidate_tools = RetrievalHelper.rank_tools_for_query(
            candidate_by_name.values(),
            query,
            devices,
        )
        return vector_results, candidate_tools

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
    def rerank_tools_for_devices(tools: Iterable[T], devices: Iterable[Any], limit: int) -> list[T]:
        """Softly prefer device-compatible tools without erasing alternatives."""
        devices = list(devices)
        tools = list(tools)
        ranked = sorted(
            enumerate(tools),
            key=lambda pair: (
                -RetrievalHelper.tool_device_compatibility(pair[1], devices),
                pair[0],
            ),
        )
        return [tool for _, tool in ranked[:limit]]
