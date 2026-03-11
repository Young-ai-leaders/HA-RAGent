from typing import Any, Dict, List
from abc import ABC, abstractmethod

from homeassistant.core import HomeAssistant

from ...models.device import Device
from ...models.device_embedding import DeviceEmbedding

class ABaseDbBackend(ABC):
    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]):
        self.hass = hass
        self.client_options = client_options

    @staticmethod
    @abstractmethod
    def get_name(client_options: dict[str, Any]):
        raise NotImplementedError()
    
    @staticmethod
    @abstractmethod
    async def async_validate_connection(hass: HomeAssistant, user_input: Dict[str, Any]) -> str | None:
        raise NotImplementedError()

    @abstractmethod
    async def async_cleanup_database(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    async def async_reset_database(self, config_subentry: dict, collection_name: str, embedding_length: int) -> None:
        raise NotImplementedError()
    
    @abstractmethod
    async def async_save_device_embeddings(self, config_subentry: dict, collection_name: str, device_embeddings: List[DeviceEmbedding]) -> None:
        raise NotImplementedError()
    
    @abstractmethod
    async def async_retrieve_devices(self, config_subentry: dict, collection_name: str, query_embedding: List[float], top_k: int = 10) -> List[Device]:
        raise NotImplementedError()
