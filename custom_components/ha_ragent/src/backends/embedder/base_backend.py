from typing import Any, Dict, List
from abc import ABC, abstractmethod

from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigSubentry

from ...models.device import Device
from ...models.device_embedding import DeviceEmbedding
from ...models.tool import LlmTool
from ...models.tool_embedding import LlmToolEmbedding


class ABaseEmbedder(ABC):
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
    async def async_embed_text(self, config_subentry: dict, text: str) -> List[float]:
        raise NotImplementedError()
    
    @abstractmethod
    async def async_embed_object(self, config_subentry: dict, devices: List[Device | LlmTool]) -> List[DeviceEmbedding | LlmToolEmbedding]:
        raise NotImplementedError()