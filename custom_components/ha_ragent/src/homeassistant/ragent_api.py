from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from custom_components.ha_ragent.src.const import (
    RAGENT_LLM_API_ID,
    RAGENT_LLM_API_NAME
)

from custom_components.ha_ragent.src.homeassistant.tools.planned_action import RAGentPlannedActionTool
from custom_components.ha_ragent.src.homeassistant.tools.clear_planned_actions import RAGentClearPlannedActionsTool
from custom_components.ha_ragent.src.homeassistant.tools.search_tools import RAGentSemanticSearchTool


def resolve_llm_api_id(api_id: str) -> str:
    """Route the built-in Assist selection through the HA-RAGent API."""
    assist_api_id = getattr(llm, "LLM_API_ASSIST", "assist")
    return RAGENT_LLM_API_ID if api_id == assist_api_id else api_id


class RAGentAugmentedAPIInstance(llm.APIInstance):
    def __init__(
        self,
        hass: HomeAssistant,
        wrapped_api: llm.APIInstance,
        entry_id: str,
        subentry_id: str,
        llm_context: llm.LLMContext,
        agent_id: str
    ) -> None:
        self.hass = hass
        self._wrapped_api = wrapped_api
        self.prompt = getattr(wrapped_api, "prompt", "")
        self.custom_serializer = getattr(wrapped_api, "custom_serializer", None)

        wrapped_tools = list(getattr(wrapped_api, "tools", []) or [])
        scoped_search_tool = RAGentSemanticSearchTool(hass, entry_id, subentry_id)
        planned_action_tool = RAGentPlannedActionTool(
            hass,
            subentry_id=subentry_id,
            agent_id=agent_id,
            context=llm_context.context,
            language=llm_context.language,
            device_id=llm_context.device_id
        )
        clear_planned_actions_tool = RAGentClearPlannedActionsTool(hass, subentry_id)
        self.tools = []
        for tool in wrapped_tools:
            tool_name = getattr(tool, "name", None)
            if tool_name == RAGentSemanticSearchTool.name:
                self.tools.append(scoped_search_tool)
            elif tool_name == RAGentPlannedActionTool.name:
                self.tools.append(planned_action_tool)
            elif tool_name == RAGentClearPlannedActionsTool.name:
                self.tools.append(clear_planned_actions_tool)
            else:
                self.tools.append(tool)
        if not any(getattr(tool, "name", None) == RAGentSemanticSearchTool.name for tool in wrapped_tools):
            self.tools.append(scoped_search_tool)
        if not any(
            getattr(tool, "name", None) == RAGentPlannedActionTool.name
            for tool in wrapped_tools
        ):
            self.tools.append(planned_action_tool)
        if not any(
            getattr(tool, "name", None) == RAGentClearPlannedActionsTool.name
            for tool in wrapped_tools
        ):
            self.tools.append(clear_planned_actions_tool)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped API instance."""
        return getattr(self._wrapped_api, name)

    def set_conversation_agent_id(self, agent_id: str) -> None:
        """Bind delayed actions to the conversation entity handling this turn."""
        for tool in self.tools:
            if isinstance(tool, RAGentPlannedActionTool):
                tool.agent_id = agent_id

    def set_search_scope(self, entry_id: str, subentry_id: str) -> None:
        """Bind semantic search to the RAGent configuration handling this turn."""
        for tool in self.tools:
            if isinstance(tool, RAGentSemanticSearchTool):
                tool.entry_id = entry_id
                tool.subentry_id = subentry_id

    async def async_call_tool(self, tool_input: llm.ToolInput) -> Any:
        """Intercept calls to RAGent tools and delegate to the appropriate tool instance."""
        if tool_input.tool_name in {
            RAGentSemanticSearchTool.name,
            RAGentPlannedActionTool.name,
            RAGentClearPlannedActionsTool.name,
        }:
            for tool in self.tools:
                if tool.name == tool_input.tool_name:
                    return await tool.async_call(tool_input)

        return await self._wrapped_api.async_call_tool(tool_input)

class RAGentLLMAPI(llm.API):
    def __init__(self, hass: HomeAssistant) -> None:
        self.hass = hass
        self.id = RAGENT_LLM_API_ID
        self.name = RAGENT_LLM_API_NAME

    async def async_get_api_instance(self, llm_context: llm.LLMContext, *args: Any, **kwargs: Any) -> llm.APIInstance:
        assist_api = await llm.async_get_api(self.hass, getattr(llm, "LLM_API_ASSIST", "assist"), llm_context=llm_context)
        return RAGentAugmentedAPIInstance(
            self.hass,
            assist_api,
            entry_id="",
            subentry_id="",
            llm_context=llm_context,
            agent_id=None
        )
