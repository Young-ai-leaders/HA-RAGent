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
        self._contextual_query = ""
        self._candidate_context: list[dict[str, object]] = []

    @staticmethod
    def _clean(value: object) -> str:
        return " ".join(str(value or "").split())

    @classmethod
    def _build_search_query(cls, latest_request: str = "", area: str = "", floor: str = "") -> str:
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

        return "\n".join(sections)[:MAX_SEARCH_QUERY_CHARS].strip()

    def set_search_context(
        self,
        latest_request: str,
        area: str,
        floor: str,
        candidates: list[dict[str, object]]
    ) -> None:
        """Bind trusted context for the current conversation turn."""
        self._contextual_query = self._build_search_query(
            latest_request=latest_request,
            area=area,
            floor=floor,
        )
        self._candidate_context = candidates

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
            return model_search_query[:MAX_SEARCH_QUERY_CHARS]
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

    async def async_call(self, tool_input, *args, **kwargs) -> dict[str, object]:
        model_search_query = self._clean(tool_input.tool_args.get("search_query"))
        query = await self._validate_query(tool_input)
        if not query:
            return {"error": "search_query must not be empty"}
        _logger.debug(
            "Semantic search model search_query=%r effective query=%r",
            model_search_query,
            query,
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

        for entry, subentry_id, subentry, device_limit, tool_limit in self._iter_searchable_entries():
            try:
                query_embedding = await self._embed_query_for_subentry(entry, subentry, query)
            except Exception as err:
                errors.append(f"Failed to embed query for subentry {subentry.title}: {err}")
                continue

            try:
                compatible_devices: list[Device | dict[str, object]] = list(self._candidate_context)
                if search_devices and len(devices) < device_limit:
                    collection_name = f"devices_{subentry_id}"
                    candidate_limit = RetrievalHelper.adaptive_candidate_limit(device_limit)
                    scored_devices, all_devices = await asyncio.gather(
                        entry.vector_db_backend.async_retrieve_scored_objects(
                            DeviceEmbedding,
                            dict(subentry.data),
                            collection_name,
                            query_embedding,
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
                        query,
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
                        device_limit,
                        metadata_score=lambda device: RetrievalHelper.field_match_score(
                            query,
                            (
                                device.area_name,
                                device.floor_name,
                                *(device.domain or []),
                                device.device_class,
                                *(device.device_labels or []),
                            ),
                        ),
                    )
                    compatible_devices = retrieved_devices
                    for device in retrieved_devices:
                        if not isinstance(device, Device) or device.id in seen_device_ids:
                            continue
                        seen_device_ids.add(device.id)
                        state = self.hass.states.get(device.id)
                        devices.append(
                            {
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
                                    else None
                                ),
                            }
                        )
                        if len(devices) >= device_limit:
                            break

                if search_tools and len(tools) < tool_limit:
                    collection_name = f"tools_{subentry_id}"
                    candidate_limit = RetrievalHelper.adaptive_candidate_limit(tool_limit)
                    scored_tools, all_tools = await asyncio.gather(
                        entry.vector_db_backend.async_retrieve_scored_objects(
                            LlmToolEmbedding,
                            dict(subentry.data),
                            collection_name,
                            query_embedding,
                            candidate_limit,
                        ),
                        entry.vector_db_backend.async_get_lexical_objects(
                            LlmToolEmbedding,
                            dict(subentry.data),
                            collection_name,
                        ),
                    )
                    retrieved_tools = RetrievalHelper.rank_scored_candidates(
                        scored_tools,
                        all_tools,
                        query,
                        lambda tool: tool.name,
                        lambda tool: (tool.name, *tool.canonical_name_parts, tool.family, tool.description),
                        min(candidate_limit, tool_limit * 2),
                        metadata_score=lambda tool: RetrievalHelper.field_match_score(
                            query,
                            ((tool.parameters or {}).get("properties") or {}).keys(),
                        ),
                    )
                    retrieved_tools = RetrievalHelper.rerank_tools_for_devices(
                        retrieved_tools,
                        compatible_devices,
                        tool_limit,
                    )
                    for tool in retrieved_tools:
                        if not isinstance(tool, LlmTool) or tool.name in seen_tool_names:
                            continue
                        seen_tool_names.add(tool.name)
                        tools.append(
                            {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.parameters or {},
                            }
                        )
                        if len(tools) >= tool_limit:
                            break
            except Exception as err:
                errors.append(f"Failed to search subentry {subentry.title}: {err}")

        return {
            "search_query": query,
            "scope": scope,
            "devices": devices[:device_limit] if search_devices else [],
            "tools": tools[:tool_limit] if search_tools else [],
            "error": errors,
        }
