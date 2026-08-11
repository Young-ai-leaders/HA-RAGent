from __future__ import annotations

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


class RAGentSemanticSearchTool(llm.Tool):
    """Semantic search tool over embedded devices and tools."""

    name = RAGENT_SEMANTIC_SEARCH_TOOL_NAME
    description = (
        "Resolve Home Assistant targets with semantic search. "
        "Use when the request contains a fuzzy name, natural-language reference, area, typo, category, "
        "or may match multiple devices. "
        "Use `devices` for entities, `tools` for capabilities, or `both` when needed. "
        "Do not guess when search can resolve the target. "
        "Describe the intended target and action briefly."
    )
    parameters = vol.Schema(
        {
            vol.Required("query"): str,
            vol.Optional("scope", default="both"): vol.In(["devices", "tools", "both"]),
        }
    )

    def __init__(self, hass: HomeAssistant) -> None:
        self.hass = hass

    @staticmethod
    def _get_effective_limits(entry: Any) -> tuple[int, int]:
        """Use the same configured limits as normal retrieval."""
        entry_options = getattr(entry, "options", {}) or {}
        device_limit = int(entry_options.get(CONF_NUM_DEVICES_TO_EXTRACT, DEFAULT_NUM_DEVICES_TO_EXTRACT))
        tool_limit = int(entry_options.get(CONF_NUM_TOOLS_TO_EXTRACT, DEFAULT_NUM_TOOLS_TO_EXTRACT))
        return device_limit, tool_limit

    async def _validate_query(self, tool_input: llm.ToolInput) -> str | None:
        query = str(tool_input.tool_args.get("query", "")).strip()
        return query or None

    def _iter_searchable_entries(self):
        """Yield searchable entry and subentry combinations."""
        domain_data = self.hass.data.get(DOMAIN, {})
        for _, entry in domain_data.items():
            if not hasattr(entry, "subentries") or not hasattr(entry, "embedder_backend"):
                continue

            device_limit, tool_limit = self._get_effective_limits(entry)

            for subentry_id, subentry in entry.subentries.items():
                if subentry.data.get(CONF_LLM_HASS_API) == "none":
                    continue

                yield entry, subentry_id, subentry, device_limit, tool_limit

    async def _embed_query_for_subentry(self, entry: Any, subentry: Any, query: str) -> list[float]:
        """Embed a search query for a specific subentry."""
        return await entry.embedder_backend.async_embed_text(dict(subentry.data), query)

    async def async_call(self, tool_input, *args, **kwargs) -> dict[str, object]:
        query = await self._validate_query(tool_input)
        if not query:
            return {"error": "query must not be empty"}

        scope = str(tool_input.tool_args.get("scope", "both")).strip().lower() or "both"
        search_devices = scope in {"devices", "both"}
        search_tools = scope in {"tools", "both"}

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
                                "entity_id": device.id,
                                "name": device.name,
                                "area": device.area_name,
                                "domain": device.domain,
                                "aliases": device.aliases or [],
                                "state": state.state if state else None,
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
            "errors": errors,
        }
