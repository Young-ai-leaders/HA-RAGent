from __future__ import annotations

import asyncio
import logging
from typing import Any

import voluptuous as vol

try:
    from homeassistant.const import CONF_LLM_HASS_API
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers import llm
except ImportError:
    from custom_components.ha_ragent.src.mock import (
        CONF_LLM_HASS_API,
        MockHomeAssistant as HomeAssistant,
        llm,
    )

from custom_components.ha_ragent.src.const import (
    CONF_NUM_DEVICES_TO_EXTRACT,
    CONF_NUM_TOOLS_TO_EXTRACT,
    DEFAULT_NUM_DEVICES_TO_EXTRACT,
    DEFAULT_NUM_TOOLS_TO_EXTRACT,
    DOMAIN,
    RAGENT_SEMANTIC_SEARCH_TOOL_NAME,
)
from custom_components.ha_ragent.src.models.embedding.device import Device
from custom_components.ha_ragent.src.models.embedding.device_embedding import DeviceEmbedding
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool
from custom_components.ha_ragent.src.models.embedding.tool_embedding import LlmToolEmbedding
from custom_components.ha_ragent.src.homeassistant.helpers.retrieval_helper import RetrievalHelper
from custom_components.ha_ragent.src.utils import get_tool_description

_logger = logging.getLogger(__name__)
MAX_SEARCH_QUERY_CHARS = 4000


class RAGentSemanticSearchTool(llm.Tool):
    name = RAGENT_SEMANTIC_SEARCH_TOOL_NAME
    parameters = vol.Schema(
        {
            vol.Required("search_query"): str,
            vol.Optional("scope", default="devices_and_tools"): vol.In(["devices", "tools", "devices_and_tools"]),
        }
    )

    def __init__(self, hass: HomeAssistant, entry_id: str, subentry_id: str, language: str | None = None) -> None:
        self.hass = hass
        self.entry_id = entry_id
        self.subentry_id = subentry_id
        self.description = get_tool_description(language, RAGENT_SEMANTIC_SEARCH_TOOL_NAME)
        self._latest_request = ""
        self._contextual_query = ""
        self._candidate_context: list[dict[str, object]] = []

    @staticmethod
    def _clean(value: object) -> str:
        return " ".join(str(value or "").split())

    @classmethod
    def _candidate_summary(cls, candidate: dict[str, object]) -> str:
        """Return compact searchable text for a current device candidate."""
        domains = candidate.get("domain") or []
        if isinstance(domains, str):
            domains = [domains]
        values = (
            candidate.get("name"),
            candidate.get("friendly_name"),
            candidate.get("area"),
            candidate.get("floor"),
            *domains,
            candidate.get("device_class"),
            candidate.get("state"),
            candidate.get("unit_of_measurement"),
        )
        return " | ".join(cls._clean(value) for value in values if cls._clean(value))

    @classmethod
    def _build_search_query(
        cls,
        latest_request: str = "",
        area: str = "",
        floor: str = "",
        candidates: list[dict[str, object]] | None = None,
    ) -> str:
        """Build a bounded fallback query from user requests and trusted location."""
        sections: list[str] = []
        latest = cls._clean(latest_request)
        if latest:
            sections.append(f"Current request: {latest}")

        current_area = cls._clean(area)
        current_floor = cls._clean(floor)
        if current_area:
            sections.append(f"Default area when the request has no explicit location: {current_area}")
        if current_floor:
            sections.append(f"Default floor when the request has no explicit location: {current_floor}")

        for candidate in (candidates or [])[:8]:
            summary = cls._candidate_summary(candidate)
            if summary:
                sections.append(f"Current candidate: {summary}")

        return "\n".join(sections)[:MAX_SEARCH_QUERY_CHARS].strip()

    def set_search_context(
        self,
        *,
        latest_request: str = "",
        area: str = "",
        floor: str = "",
        candidates: list[dict[str, object]] | None = None,
    ) -> None:
        """Bind trusted context for the current conversation turn."""
        self._latest_request = self._clean(latest_request)
        self._contextual_query = self._build_search_query(
            latest_request=latest_request,
            area=area,
            floor=floor,
            candidates=candidates,
        )
        self._completed_candidate_names: set[str] = set()
        self._candidate_context = list(candidates or [])

    def refresh_candidates(self, candidates: list[dict[str, object]]) -> None:
        """Replace candidate context after a corrective search."""
        completed = getattr(self, "_completed_candidate_names", set())
        self._candidate_context = [
            candidate
            for candidate in candidates
            if str(candidate.get("name", "")).casefold() not in completed
        ]

    def prune_candidates(self, completed_names: set[str]) -> None:
        """Remove completed targets from later corrective searches."""
        normalized = {name.casefold() for name in completed_names}
        self._completed_candidate_names = getattr(self, "_completed_candidate_names", set()) | normalized
        self._candidate_context = [
            candidate
            for candidate in self._candidate_context
            if str(candidate.get("name", "")).casefold() not in self._completed_candidate_names
        ]

    @staticmethod
    def _get_effective_limits(entry: Any, subentry: Any) -> tuple[int, int]:
        """Use the same configured limits as normal retrieval."""
        entry_options = getattr(entry, "options", {}) or {}
        subentry_data = getattr(subentry, "data", {}) or {}
        runtime_options = {**entry_options, **subentry_data}
        device_limit = int(runtime_options.get(CONF_NUM_DEVICES_TO_EXTRACT, DEFAULT_NUM_DEVICES_TO_EXTRACT))
        tool_limit = int(runtime_options.get(CONF_NUM_TOOLS_TO_EXTRACT, DEFAULT_NUM_TOOLS_TO_EXTRACT))
        return device_limit, tool_limit

    async def _validate_query(self, tool_input: llm.ToolInput) -> str | None:
        """Prefer the model's explicit search intent, with user context as fallback."""
        model_search_query = self._clean(tool_input.tool_args.get("search_query"))
        if model_search_query:
            sections = [f"Search intent: {model_search_query}"]
            if self._contextual_query:
                sections.append(self._contextual_query)
            return "\n".join(sections)[:MAX_SEARCH_QUERY_CHARS].strip()
        return self._contextual_query or None

    def _iter_searchable_entries(self):
        """Yield only the active entry and subentry."""
        domain_data = self.hass.data.get(DOMAIN, {})
        entry = domain_data.get(self.entry_id)
        if not entry or not hasattr(entry, "subentries") or not hasattr(entry, "embedder_backend"):
            return
        subentry = entry.subentries.get(self.subentry_id)
        if not subentry or subentry.data.get(CONF_LLM_HASS_API) == "none":
            return
        device_limit, tool_limit = self._get_effective_limits(entry, subentry)
        yield entry, self.subentry_id, subentry, device_limit, tool_limit

    async def _embed_query_for_subentry(self, entry: Any, subentry: Any, query: str) -> list[float]:
        """Embed a search query for a specific subentry."""
        return await entry.embedder_backend.async_embed_text(dict(subentry.data), query)

    def _device_search_query(self, model_search_query: str, fallback_query: str) -> str:
        """Use trusted request text for devices instead of a model rewrite."""
        return self._latest_request or model_search_query or fallback_query

    def _tool_search_query(
        self,
        model_search_query: str,
        devices: list[Device | dict[str, object]],
    ) -> str:
        """Use only canonical action and resolved domains for tool retrieval."""
        return RetrievalHelper.build_tool_search_query(
            self._latest_request,
            model_search_query,
            devices,
        )

    @staticmethod
    def _tool_search_feedback(
        search_tools: bool,
        confidence: str,
        tools: list[dict[str, object]],
    ) -> tuple[str, bool, str]:
        """Return an explicit status without discarding uncertain candidates."""
        if not search_tools:
            return "not_requested", False, ""
        if not tools:
            return (
                "no_tools_found",
                True,
                "No tools were retrieved. Do not invent a tool name; "
                "broaden retrieval or report that no action tool is available.",
            )
        if confidence == "low":
            return (
                "weak_candidates",
                True,
                "Candidate confidence is weak. Reuse these candidates, "
                "broaden retrieval if needed, and do not invent a tool.",
            )
        return "candidates_found", False, ""

    def _device_candidate(self, device: Device) -> dict[str, object]:
        """Return enough current device data to answer without another search."""
        state = self.hass.states.get(device.id)
        return {
            "name": device.id,
            "friendly_name": device.friendly_name,
            "area": device.area_name,
            "floor": device.floor_name,
            "domain": device.domain,
            "device_class": device.device_class,
            "aliases": device.aliases or [],
            "state": state.state if state else None,
            "unit_of_measurement": (
                state.attributes.get("unit_of_measurement")
                if state
                else device.unit_of_measurement
            ),
        }

    async def async_call(self, tool_input, *args, **kwargs) -> dict[str, object]:
        model_search_query = self._clean(tool_input.tool_args.get("search_query"))
        query = await self._validate_query(tool_input)
        if not query:
            return {"error": "search_query must not be empty"}
        device_query = self._device_search_query(model_search_query, query)
        _logger.debug(
            "Semantic search model query=%r trusted device query=%r",
            model_search_query,
            device_query,
        )

        scope = (
            str(tool_input.tool_args.get("scope", "devices_and_tools")).strip().lower()
            or "devices_and_tools"
        )
        search_devices = scope in {"devices", "devices_and_tools"}
        search_tools = scope in {"tools", "devices_and_tools"}

        devices: list[dict[str, object]] = []
        tools: list[dict[str, object]] = []
        errors: list[str] = []
        seen_device_ids: set[str] = set()
        seen_tool_names: set[str] = set()
        device_limit = 0
        tool_limit = 0
        tool_query = ""
        tool_confidence = "not_requested"

        for entry, subentry_id, subentry, device_limit, tool_limit in self._iter_searchable_entries():
            try:
                compatible_devices: list[Device | dict[str, object]] = list(self._candidate_context)
                if search_devices and len(devices) < device_limit:
                    device_embedding = await self._embed_query_for_subentry(
                        entry,
                        subentry,
                        device_query,
                    )
                    collection_name = f"devices_{subentry_id}"
                    candidate_limit = RetrievalHelper.adaptive_candidate_limit(device_limit)
                    scored_devices, all_devices = await asyncio.gather(
                        entry.vector_db_backend.async_retrieve_scored_objects(
                            DeviceEmbedding,
                            dict(subentry.data),
                            collection_name,
                            device_embedding,
                            candidate_limit,
                        ),
                        entry.vector_db_backend.async_get_lexical_objects(
                            DeviceEmbedding,
                            dict(subentry.data),
                            collection_name,
                        ),
                    )
                    retrieved_devices = RetrievalHelper.rank_scored_candidates(
                        scored_devices,
                        all_devices,
                        device_query,
                        lambda device: device.id,
                        lambda device: (
                            device.id,
                            device.friendly_name,
                            *(device.aliases or []),
                            device.area_name,
                            device.floor_name,
                            *(device.domain or []),
                            device.device_class,
                            *(device.device_labels or []),
                        ),
                        candidate_limit,
                        metadata_score=lambda device: 2.0 * RetrievalHelper.device_target_score(
                            device_query,
                            device,
                        ),
                        trim_confident=False,
                    )
                    retrieved_devices = RetrievalHelper.select_device_candidates(
                        device_query,
                        retrieved_devices,
                        device_limit,
                    )
                    if retrieved_devices:
                        compatible_devices = retrieved_devices
                        self.refresh_candidates([
                            self._device_candidate(device)
                            for device in retrieved_devices
                            if isinstance(device, Device)
                        ])
                    for device in retrieved_devices:
                        if not isinstance(device, Device) or device.id in seen_device_ids:
                            continue
                        seen_device_ids.add(device.id)
                        devices.append(self._device_candidate(device))

                if search_tools and len(tools) < tool_limit:
                    tool_query = self._tool_search_query(
                        model_search_query,
                        compatible_devices,
                    )
                    tool_embedding = await self._embed_query_for_subentry(
                        entry,
                        subentry,
                        tool_query,
                    )
                    collection_name = f"tools_{subentry_id}"
                    candidate_limit = RetrievalHelper.adaptive_candidate_limit(tool_limit)
                    scored_tools, all_tools = await asyncio.gather(
                        entry.vector_db_backend.async_retrieve_scored_objects(
                            LlmToolEmbedding,
                            dict(subentry.data),
                            collection_name,
                            tool_embedding,
                            candidate_limit,
                        ),
                        entry.vector_db_backend.async_get_lexical_objects(
                            LlmToolEmbedding,
                            dict(subentry.data),
                            collection_name,
                        ),
                    )
                    scored_tools, candidate_tools = RetrievalHelper.build_tool_candidate_pool(
                        scored_tools,
                        all_tools,
                        tool_query,
                        compatible_devices,
                    )
                    semantic_ranks = {
                        result.item.name: result.rank
                        for result in scored_tools
                    }
                    semantic_scores = {
                        result.item.name: result.score
                        for result in scored_tools
                    }
                    retrieved_tools = RetrievalHelper.rank_tool_candidates(
                        scored_tools,
                        candidate_tools,
                        tool_query,
                        compatible_devices,
                        candidate_limit,
                    )
                    tool_confidence = RetrievalHelper.tool_search_confidence(
                        retrieved_tools,
                        tool_query,
                        compatible_devices,
                    )
                    result_tool_limit = tool_limit
                    if tool_confidence == "low":
                        result_tool_limit = min(
                            candidate_limit,
                            max(tool_limit * 2, tool_limit + 2),
                        )
                    for tool in retrieved_tools:
                        if not isinstance(tool, LlmTool) or tool.name in seen_tool_names:
                            continue
                        seen_tool_names.add(tool.name)
                        ranking_signals = RetrievalHelper.tool_ranking_signals(
                            tool,
                            tool_query,
                            compatible_devices,
                            semantic_rank=semantic_ranks.get(tool.name),
                            semantic_score=semantic_scores.get(tool.name),
                        )
                        tools.append(
                            {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.parameters or {},
                                "metadata": tool.metadata.to_dict() if tool.metadata else None,
                                "canonical_action": tool.canonical_action,
                                "supported_domains": tool.canonical_supported_domains,
                                "ranking_signals": {
                                    name: round(value, 4)
                                    for name, value in ranking_signals.items()
                                },
                                "retrieval_score": round(
                                    RetrievalHelper.tool_signal_score(ranking_signals),
                                    4,
                                ),
                            }
                        )
                        if len(tools) >= result_tool_limit:
                            break
            except Exception as err:
                errors.append(f"Failed to search subentry {subentry.title}: {err}")

        returned_devices = RetrievalHelper.select_device_candidates(
            device_query,
            devices,
            device_limit,
        )
        if not returned_devices:
            returned_devices = list(self._candidate_context[:8])
        tool_status, fallback_required, tool_message = self._tool_search_feedback(
            search_tools,
            tool_confidence,
            tools,
        )
        return {
            "result_type": "candidate_search",
            "candidate_notice": "Candidates only; no action has been performed.",
            "candidate_data_notice": (
                "Use the included state and location data directly when it answers the request."
            ),
            "search_query": query,
            "device_search_query": device_query if search_devices else "",
            "tool_search_query": tool_query if search_tools else "",
            "scope": scope,
            "candidate_devices": returned_devices,
            "candidate_tools": tools if search_tools else [],
            "tool_search_status": tool_status,
            "tool_search_confidence": tool_confidence,
            "fallback_required": fallback_required,
            "tool_search_message": tool_message,
            "error": errors,
        }
