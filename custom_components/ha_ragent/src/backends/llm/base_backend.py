from abc import ABC, abstractmethod
from typing import Any, Dict, List, AsyncGenerator
import aiohttp

from homeassistant.helpers.aiohttp_client import async_get_clientsession

from homeassistant.core import HomeAssistant
from homeassistant.helpers.llm import APIInstance

from custom_components.ha_ragent.src.models.tool import LlmTool

from ...const import (
    CONF_LLM_HOST,
    CONF_LLM_PORT,
    CONF_LLM_SSL,
)

class ALlmBaseBackend(ABC):
    _default_timeout = aiohttp.ClientTimeout(total=5)
    _chat_timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)

    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]):
        self.hass = hass
        self.client_options = client_options

        self._url_base = {
            "hostname": client_options.get(CONF_LLM_HOST),
            "port": client_options.get(CONF_LLM_PORT),
            "ssl": client_options.get(CONF_LLM_SSL),
        }

        self._api_key = str(client_options.get("api_key", "") or "").strip()
    
    @staticmethod
    def format_url(hostname: str, port: str, ssl: bool, path: str) -> str:
        return f"{'https' if ssl else 'http'}://{hostname}{ ':' + str(port) if port else ''}{path}"

    @staticmethod
    def get_name(client_options: dict[str, Any]):
        raise NotImplementedError()
    
    @staticmethod
    async def async_validate_connection(hass: HomeAssistant, user_input: Dict[str, Any]) -> str | None:
        raise NotImplementedError()

    @staticmethod
    def convert_tools_to_model_format(tools: List[LlmTool]) -> List[Dict[str, Any]]:
        return [tool.to_tool_dict() for tool in tools]

    @abstractmethod
    async def async_get_model_info(self, model_name: str) -> Dict[str, Any]:
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