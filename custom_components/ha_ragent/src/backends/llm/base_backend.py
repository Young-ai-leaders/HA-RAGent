from abc import ABC, abstractmethod
from typing import Any, Dict, List, AsyncGenerator

from homeassistant.core import HomeAssistant
from homeassistant.helpers.llm import APIInstance

from custom_components.ha_ragent.src.models.tool import LlmTool

class ALlmBaseBackend(ABC):
    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]):
        self.hass = hass
        self.client_options = client_options
    
    @staticmethod
    def _format_url(hostname: str, port: str, ssl: bool, path: str) -> str:
        return f"{'https' if ssl else 'http'}://{hostname}{ ':' + str(port) if port else ''}{path}"

    @staticmethod
    def get_name(client_options: dict[str, Any]):
        raise NotImplementedError()
    
    @staticmethod
    async def async_validate_connection(hass: HomeAssistant, user_input: Dict[str, Any]) -> str | None:
        raise NotImplementedError()
    
    @abstractmethod
    async def async_preload_model(self, config_subentry: dict) -> None:
        raise NotImplementedError()
    
    @abstractmethod
    async def async_unload_model(self, config_subentry: dict) -> None:
        raise NotImplementedError()
    
    @abstractmethod
    async def async_get_available_models(self) -> List[str]:
        raise NotImplementedError()
    
    @abstractmethod
    async def async_send_chat_request(self, config_subentry: dict, messages: List[Dict[str, str]], tools: List[LlmTool]) -> AsyncGenerator[str, None]:
        raise NotImplementedError()