from typing import Any, Dict, List
from abc import ABC, abstractmethod

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigSubentry

from custom_components.ha_ragent.src.const import (
    CONF_EMBEDDING_API_KEY,
    CONF_EMBEDDING_HOST,
    CONF_EMBEDDING_PORT,
    CONF_EMBEDDING_SSL,
)

from ...models.device import Device
from ...models.device_embedding import DeviceEmbedding
from ...models.tool import LlmTool
from ...models.tool_embedding import LlmToolEmbedding


class ABaseEmbedder(ABC):
    _default_timeout = aiohttp.ClientTimeout(total=5)
    _chat_timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)

    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]):
        self._hass = hass
        self._client_options = client_options
        self._api_key = str(client_options.get(CONF_EMBEDDING_API_KEY, "") or "").strip()

        self._url_base = {
            "hostname": client_options.get(CONF_EMBEDDING_HOST),
            "port": client_options.get(CONF_EMBEDDING_PORT),
            "ssl": client_options.get(CONF_EMBEDDING_SSL),
        }
    
    @staticmethod
    def format_url(hostname: str, port: str, ssl: bool, path: str) -> str:
        return f"{'https' if ssl else 'http'}://{hostname}{ ':' + str(port) if port else ''}{path}"

    @staticmethod
    def get_name() -> str:
        return "Embedder"
    
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
    async def async_embed_text(self, config_subentry: dict, text: str) -> List[float]:
        raise NotImplementedError()
    
    @abstractmethod
    async def async_embed_object(self, object_type: type[DeviceEmbedding | LlmToolEmbedding], config_subentry: dict, devices: List[Device | LlmTool]) -> List[DeviceEmbedding | LlmToolEmbedding]:
        raise NotImplementedError()
