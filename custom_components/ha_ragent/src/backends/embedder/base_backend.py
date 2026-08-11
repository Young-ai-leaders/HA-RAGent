from abc import ABC, abstractmethod
import aiohttp
from typing import Any, Dict, List

from custom_components.ha_ragent.src.models.model_info import ModelInfo

try:
    from homeassistant.core import HomeAssistant
except ImportError:
    from custom_components.ha_ragent.src.backends.mock import MockHomeAssistant as HomeAssistant

from custom_components.ha_ragent.src.const import (
    CONF_EMBEDDING_API_KEY,
    CONF_EMBEDDING_HOST,
    CONF_EMBEDDING_PORT,
    CONF_EMBEDDING_SSL
)
from custom_components.ha_ragent.src.models.device import Device
from custom_components.ha_ragent.src.models.device_embedding import DeviceEmbedding
from custom_components.ha_ragent.src.models.tool import LlmTool
from custom_components.ha_ragent.src.models.tool_embedding import LlmToolEmbedding


class ABaseEmbedder(ABC):
    _default_timeout = aiohttp.ClientTimeout(total=5)
    _chat_timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)

    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]):
        self._hass = hass
        self._client_options = client_options
        self._api_key = ABaseEmbedder.normalize_api_key(client_options.get(CONF_EMBEDDING_API_KEY))
        self._url_base = {
            "hostname": client_options.get(CONF_EMBEDDING_HOST),
            "port": client_options.get(CONF_EMBEDDING_PORT),
            "ssl": client_options.get(CONF_EMBEDDING_SSL),
        }

    @staticmethod
    def normalize_api_key(api_key: str) -> str:
        return str(api_key or "").strip()
        
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
    async def async_get_model_info(self, model_name: str) -> ModelInfo:
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
    async def async_embed_text(self, config_subentry: dict, text: str, **kwargs) -> List[float]:
        raise NotImplementedError()
    
    @abstractmethod
    async def async_embed_object(self, config_subentry: dict, objects: List[Device | LlmTool]) -> List[DeviceEmbedding | LlmToolEmbedding]:
        raise NotImplementedError()
