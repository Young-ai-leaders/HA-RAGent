from __future__ import annotations

from typing import Any, List

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from custom_components.ha_ragent.src.const import (
    DOMAIN,
    RAGENT_LLM_API_ID,
    RAGENT_LLM_API_NAME
)

from custom_components.ha_ragent.src.homeassistant.tools.planned_action import RAGentPlannedActionTool
from custom_components.ha_ragent.src.homeassistant.tools.cancel_all_planned_actions import RAGentCancelAllPlannedActionsTool
from custom_components.ha_ragent.src.homeassistant.tools.list_planned_actions import RAGentListPlannedActionsTool
from custom_components.ha_ragent.src.homeassistant.tools.search_tools import RAGentSemanticSearchTool
from custom_components.ha_ragent.src.homeassistant.tools.forget_fact import RAGentForgetTool
from custom_components.ha_ragent.src.homeassistant.tools.remember_fact import RAGentRememberTool


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
        self._custom_tool_namespace = DOMAIN

        wrapped_tools = list(getattr(wrapped_api, "tools", []) or [])
        scoped_search_tool = RAGentSemanticSearchTool(hass, entry_id, subentry_id, llm_context.language)
        planned_action_tool = RAGentPlannedActionTool(
            hass,
            subentry_id=subentry_id,
            agent_id=agent_id,
            context=llm_context.context,
            language=llm_context.language,
            device_id=llm_context.device_id
        )
        cancel_all_planned_actions_tool = RAGentCancelAllPlannedActionsTool(hass, subentry_id, llm_context.language)
        list_planned_actions_tool = RAGentListPlannedActionsTool(hass, subentry_id, llm_context.language)
        remember_tool = RAGentRememberTool(hass, entry_id, subentry_id, llm_context.language)
        forget_tool = RAGentForgetTool(hass, entry_id, subentry_id, llm_context.language)
        self.tools = []
        for tool in wrapped_tools:
            tool_name = getattr(tool, "name", None)
            if tool_name == RAGentSemanticSearchTool.name:
                self.tools.append(scoped_search_tool)
            elif tool_name == RAGentPlannedActionTool.name:
                self.tools.append(planned_action_tool)
            elif tool_name == RAGentCancelAllPlannedActionsTool.name:
                self.tools.append(cancel_all_planned_actions_tool)
            elif tool_name == RAGentListPlannedActionsTool.name:
                self.tools.append(list_planned_actions_tool)
            elif tool_name == RAGentRememberTool.name:
                self.tools.append(remember_tool)
            elif tool_name == RAGentForgetTool.name:
                self.tools.append(forget_tool)
            else:
                self.tools.append(tool)

        if RAGentAugmentedAPIInstance._check_if_tool_exists(RAGentSemanticSearchTool.name, wrapped_tools):
            self.tools.append(scoped_search_tool)
        if RAGentAugmentedAPIInstance._check_if_tool_exists(RAGentPlannedActionTool.name, wrapped_tools):
            self.tools.append(planned_action_tool)
        if RAGentAugmentedAPIInstance._check_if_tool_exists(RAGentCancelAllPlannedActionsTool.name, wrapped_tools):
            self.tools.append(cancel_all_planned_actions_tool)
        if RAGentAugmentedAPIInstance._check_if_tool_exists(RAGentListPlannedActionsTool.name, wrapped_tools):
            self.tools.append(list_planned_actions_tool)
        if RAGentAugmentedAPIInstance._check_if_tool_exists(RAGentRememberTool.name, wrapped_tools):
            self.tools.append(remember_tool)
        if RAGentAugmentedAPIInstance._check_if_tool_exists(RAGentForgetTool.name, wrapped_tools):
            self.tools.append(forget_tool)

        for tool in self.tools:
            if isinstance(tool, (RAGentSemanticSearchTool, RAGentPlannedActionTool, RAGentCancelAllPlannedActionsTool)):
                tool.name = f"{self._custom_tool_namespace}__{tool.name}"

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped API instance."""
        return getattr(self._wrapped_api, name)

    @staticmethod
    def _check_if_tool_exists(tool_name: str, tool_list: List) -> bool:
        return not any(getattr(tool, "name", None) == tool_name for tool in tool_list)

    def set_conversation_agent_id(self, agent_id: str) -> None:
        """Bind delayed actions to the conversation entity handling this turn."""
        for tool in self.tools:
            if isinstance(tool, RAGentPlannedActionTool):
                tool.agent_id = agent_id

    def set_search_scope(self, entry_id: str, subentry_id: str) -> None:
        """Bind scoped custom tools to the RAGent configuration handling this turn."""
        for tool in self.tools:
            if isinstance(tool, RAGentSemanticSearchTool):
                tool.entry_id = entry_id
                tool.subentry_id = subentry_id
            elif isinstance(tool, (RAGentRememberTool, RAGentForgetTool)):
                tool.entry_id = entry_id
                tool.subentry_id = subentry_id

    async def async_call_tool(self, tool_input: llm.ToolInput) -> Any:
        """Intercept calls to RAGent tools and delegate to the appropriate tool instance."""
        custom_tool_names = {
            f"{self._custom_tool_namespace}__{RAGentSemanticSearchTool.name}",
            f"{self._custom_tool_namespace}__{RAGentPlannedActionTool.name}",
            f"{self._custom_tool_namespace}__{RAGentCancelAllPlannedActionsTool.name}",
            f"{self._custom_tool_namespace}__{RAGentListPlannedActionsTool.name}",
            f"{self._custom_tool_namespace}__{RAGentRememberTool.name}",
            f"{self._custom_tool_namespace}__{RAGentForgetTool.name}",
        }
        if tool_input.tool_name in custom_tool_names:
            for tool in self.tools:
                if tool.name == tool_input.tool_name:
                    return await tool.async_call(
                        llm.ToolInput(
                            tool_name=tool_input.tool_name.rsplit("__", 1)[-1],
                            tool_args=tool_input.tool_args,
                            id=tool_input.id,
                            external=tool_input.external,
                        )
                    )

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
