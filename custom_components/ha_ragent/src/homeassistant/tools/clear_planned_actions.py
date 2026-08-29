from __future__ import annotations

import voluptuous as vol

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from custom_components.ha_ragent.src.const import (
    DOMAIN,
    RAGENT_CLEAR_PLANNED_ACTIONS_TOOL_NAME,
    RAGENT_SCHEDULED_ACTION_CANCELLERS,
)


class RAGentClearPlannedActionsTool(llm.Tool):
    name = RAGENT_CLEAR_PLANNED_ACTIONS_TOOL_NAME
    description = (
        "Cancel all currently scheduled one-time Home Assistant actions. "
        "Use this when the user asks to clear, cancel or delete all planned actions."
    )
    parameters = vol.Schema({})

    def __init__(self, hass: HomeAssistant, subentry_id: str, language: str | None = None) -> None:
        self.hass = hass
        self.subentry_id = subentry_id
        if language == "de":
            self.description = (
                "Brich alle derzeit geplanten einmaligen Home-Assistant-Aktionen ab. "
                "Verwende dieses Tool, wenn der Nutzer alle geplanten Aktionen löschen "
                "oder abbrechen möchte."
            )

    async def async_call(self, _tool_input: llm.ToolInput, *args, **kwargs) -> dict[str, object]:
        domain_data = self.hass.data.setdefault(DOMAIN, {})
        subentry_data = domain_data.setdefault(self.subentry_id, {})
        cancellers = subentry_data.get(RAGENT_SCHEDULED_ACTION_CANCELLERS, set())
        subentry_data[RAGENT_SCHEDULED_ACTION_CANCELLERS] = set()
        for cancel in cancellers:
            cancel()
        return {"success": True, "cleared": len(cancellers)}
