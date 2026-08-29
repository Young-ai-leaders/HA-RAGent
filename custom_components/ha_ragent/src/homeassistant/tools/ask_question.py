from __future__ import annotations

import voluptuous as vol

from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from custom_components.ha_ragent.src.const import RAGENT_ASK_QUESTION_TOOL_NAME


class RAGentAskQuestionTool(llm.Tool):
    name = RAGENT_ASK_QUESTION_TOOL_NAME
    description = (
        "Ask the user for clarification when the request is ambiguous or required "
        "information is missing. Use this instead of guessing or executing an action "
        "with an uncertain target. The text must be one concise question."
    )
    parameters = vol.Schema({vol.Required("text"): str})

    def __init__(self, hass: HomeAssistant, language: str | None = None) -> None:
        self.hass = hass
        if language == "de":
            self.description = (
                "Stelle dem Nutzer eine Rückfrage, wenn die Anfrage mehrdeutig ist oder "
                "Informationen fehlen. Verwende dieses Tool statt zu raten oder eine "
                "Aktion mit unsicherem Ziel auszuführen. text muss eine kurze Frage sein."
            )

    async def async_call(self, tool_input: llm.ToolInput, *args, **kwargs) -> dict[str, object]:
        text = str(tool_input.tool_args.get("text", "")).strip()
        return {"success": bool(text), "text": text}
