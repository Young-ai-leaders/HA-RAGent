from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

import voluptuous as vol

from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from custom_components.ha_ragent.src.const import (
    CONF_NUM_DEVICES_TO_EXTRACT,
    CONF_NUM_TOOLS_TO_EXTRACT,
    DEFAULT_NUM_DEVICES_TO_EXTRACT,
    DEFAULT_NUM_TOOLS_TO_EXTRACT,
    DOMAIN,
    RAGENT_SEMANTIC_SEARCH_TOOL_NAME,
)
from custom_components.ha_ragent.src.models.device import Device
from custom_components.ha_ragent.src.models.device_embedding import DeviceEmbedding
from custom_components.ha_ragent.src.models.tool import LlmTool
from custom_components.ha_ragent.src.models.tool_embedding import LlmToolEmbedding
from custom_components.ha_ragent.src.utils import get_tool_description

_logger = logging.getLogger(__name__)
MAX_SEARCH_QUERY_CHARS = 4000


class RAGentSemanticSearchTool(llm.Tool):
    name = RAGENT_SEMANTIC_SEARCH_TOOL_NAME
    parameters = vol.Schema(
        {
            vol.Optional("query", default=""): str,
            vol.Optional("scope", default="devices_and_tools"): vol.In(["devices", "tools", "devices_and_tools"]),
        }
    )

    def __init__(self, hass: HomeAssistant, entry_id: str, subentry_id: str, language: str | None = None) -> None:
        self.hass = hass
        self.entry_id = entry_id
        self.subentry_id = subentry_id
        self.description = get_tool_description(language, RAGENT_SEMANTIC_SEARCH_TOOL_NAME)
        self._contextual_query = ""

    @staticmethod
    def _clean(value: object) -> str:
        return " ".join(str(value or "").split())

    @classmethod
    def _list_text(cls, value: object) -> str:
        if isinstance(value, (list, tuple, set)):
            return ", ".join(text for item in value if (text := cls._clean(item)))
        return cls._clean(value)

    @classmethod
    def _build_search_query(
        cls,
        latest_request: str = "",
        recent_requests: Iterable[str] = (),
        area: str = "",
        floor: str = "",
        candidates: Iterable[Mapping[str, Any]] = (),
        model_query: str = ""
    ) -> str:
        """Build a bounded search query from trusted turn context."""
        sections: list[str] = []
        latest = cls._clean(latest_request)
        if latest:
            sections.append(f"Current request: {latest}")

        seen_requests = {latest.casefold()} if latest else set()
        recent: list[str] = []
        for request in recent_requests:
            text = cls._clean(request)
            if text and text.casefold() not in seen_requests:
                seen_requests.add(text.casefold())
                recent.append(text)
        for text in recent[-3:]:
            sections.append(f"Recent user request: {text}")

        location = ", ".join(
            part for part in (cls._clean(area), cls._clean(floor)) if part
        )
        if location:
            sections.append(f"Known location: {location}")

        for candidate in list(candidates)[:6]:
            fields = [
                ("friendly_name", cls._clean(candidate.get("friendly_name"))),
                ("aliases", cls._list_text(candidate.get("aliases"))),
                ("area", cls._clean(candidate.get("area"))),
                ("floor", cls._clean(candidate.get("floor"))),
                ("domain", cls._list_text(candidate.get("domain"))),
                ("device_class", cls._list_text(candidate.get("device_class"))),
                ("entity_id", cls._clean(candidate.get("name"))),
            ]
            details = "; ".join(f"{key}: {value}" for key, value in fields if value)
            if details:
                sections.append(f"Known candidate: {details}")

        if not sections and (fallback := cls._clean(model_query)):
            sections.append(fallback)

        return "\n".join(sections)[:MAX_SEARCH_QUERY_CHARS].strip()

    def set_search_context(
        self,
        latest_request: str,
        recent_requests: list[str],
        area: str,
        floor: str,
        candidates: list[dict[str, object]]
    ) -> None:
        """Bind trusted context for the current conversation turn."""
        self._contextual_query = self._build_search_query(
            latest_request=latest_request,
            recent_requests=recent_requests,
            area=area,
            floor=floor,
            candidates=candidates,
        )

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
        model_query = str(tool_input.tool_args.get("query", "")).strip()
        query = self._contextual_query or self._build_search_query(
            model_query=model_query
        )
        return query or None

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
        model_query = str(tool_input.tool_args.get("query", "")).strip()
        query = await self._validate_query(tool_input)
        if not query:
            return {"error": "query must not be empty"}
        _logger.debug(
            "Semantic search model query=%r effective query=%r",
            model_query,
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
                if search_devices and len(devices) < device_limit:
                    retrieved_devices = await entry.vector_db_backend.async_retrieve_objects(
                        object_type=DeviceEmbedding,
                        config_subentry=dict(subentry.data),
                        collection_name=f"devices_{subentry_id}",
                        query_embedding=query_embedding,
                        top_k=device_limit,
                    )
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
                                "device_class": device.domain,
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
                    retrieved_tools = await entry.vector_db_backend.async_retrieve_objects(
                        object_type=LlmToolEmbedding,
                        config_subentry=dict(subentry.data),
                        collection_name=f"tools_{subentry_id}",
                        query_embedding=query_embedding,
                        top_k=tool_limit,
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
            "query": query,
            "scope": scope,
            "devices": devices[:device_limit] if search_devices else [],
            "tools": tools[:tool_limit] if search_tools else [],
            "error": errors,
        }
