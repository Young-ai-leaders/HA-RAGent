from __future__ import annotations

import voluptuous as vol
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from custom_components.ha_ragent.src.const import RAGENT_REMEMBER_TOOL_NAME
from custom_components.ha_ragent.src.homeassistant.helpers.memory_manager import MemoryManager
from custom_components.ha_ragent.src.utils import get_tool_description


class RAGentRememberTool(llm.Tool):
    name = RAGENT_REMEMBER_TOOL_NAME
    parameters = vol.Schema({vol.Required("memory"): vol.All(str, vol.Length(min=1, max=1000))})

    def __init__(self, hass: HomeAssistant, entry_id: str, subentry_id: str, language: str | None = None) -> None:
        self.hass, self.entry_id, self.subentry_id = hass, entry_id, subentry_id
        self.description = get_tool_description(language, RAGENT_REMEMBER_TOOL_NAME)

    async def async_call(self, tool_input: llm.ToolInput, *args, **kwargs) -> dict[str, object]:
        content = MemoryManager.normalize_content(str(tool_input.tool_args.get("memory", "")))
        if not content:
            return {"success": False, "error": "memory must not be empty"}

        if len(content) > 1000:
            return {"success": False, "error": "memory must be at most 1000 characters"}

        memory = await MemoryManager(self.hass, self.entry_id, self.subentry_id).async_remember(content)
        if memory is None:
            return {"success": False, "error": "failed to store memory"}
        return {"success": True, "memory_id": memory.id, "memory": memory.content}
