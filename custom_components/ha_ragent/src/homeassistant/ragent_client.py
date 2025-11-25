from __future__ import annotations

from typing import List, Any

from homeassistant.components.conversation import ConversationInput, ConversationResult, ConversationEntity
from homeassistant.components.conversation.models import AbstractConversationAgent
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.exceptions import TemplateError, HomeAssistantError
from homeassistant.helpers import chat_session, intent, llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from ..models.device import Device

from ..const import (
    PERSONA_PROMPTS,
    CURRENT_DATE_PROMPT,
    DEVICES_PROMPT,
    SERVICES_PROMPT,
    TOOLS_PROMPT,
    AREA_PROMPT,
    USER_INSTRUCTION
)

type RAGentConfigEntry = ConfigEntry[RAGentClient]

class RAGentClient:
    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]) -> None:
        self.hass = hass
        
    def _get_devices(self, query: str) -> List[Device]:
        embedded_query = self.embedding.embed_text(query)
        return self.db_backend.retrieve_devices(embedded_query)
    
    def _build_prompt(self, query: str) -> str:
        pass
        
    def process_user_query(self, query: str):
        devices = self._get_devices(query)

    @staticmethod
    def build_prompt_template(selected_language: str, prompt_template_template: str):
        persona = PERSONA_PROMPTS.get(selected_language, PERSONA_PROMPTS["en"])
        current_date = CURRENT_DATE_PROMPT.get(selected_language, CURRENT_DATE_PROMPT["en"])
        devices = DEVICES_PROMPT.get(selected_language, DEVICES_PROMPT["en"])
        services = SERVICES_PROMPT.get(selected_language, SERVICES_PROMPT["en"])
        tools = TOOLS_PROMPT.get(selected_language, TOOLS_PROMPT["en"])
        area = AREA_PROMPT.get(selected_language, AREA_PROMPT["en"])
        user_instruction = USER_INSTRUCTION.get(selected_language, USER_INSTRUCTION["en"])

        prompt_template_template = prompt_template_template.replace("<persona>", persona)
        prompt_template_template = prompt_template_template.replace("<current_date>", current_date)
        prompt_template_template = prompt_template_template.replace("<devices>", devices)
        prompt_template_template = prompt_template_template.replace("<services>", services)
        prompt_template_template = prompt_template_template.replace("<tools>", tools)
        prompt_template_template = prompt_template_template.replace("<area>", area)
        prompt_template_template = prompt_template_template.replace("<user_instruction>", user_instruction)

        return prompt_template_template