from typing import Any, Dict, List
from abc import ABC, abstractmethod

from homeassistant.core import HomeAssistant

from ..models.device import Device
from ..models.device_embedding import DeviceEmbedding

class ABaseDbBackend(ABC):
    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]):
        self.hass = hass
        self.client_options = client_options
    
    @staticmethod
    def _format_url(username: str, password: str, hostname: str, port: str, ssl: bool) -> str:
        raise NotImplementedError()

    @staticmethod
    def get_name(client_options: dict[str, Any]):
        raise NotImplementedError()
    
    @staticmethod
    async def async_validate_connection(hass: HomeAssistant, user_input: Dict[str, Any]) -> str | None:
        raise NotImplementedError()

    @abstractmethod
    def cleanup_database(self, embedding_length: int) -> None:
        pass
    
    @abstractmethod
    def save_device_embeddings(self, device_embeddings: List[DeviceEmbedding]) -> None:
        pass
    
    @abstractmethod
    def retrieve_devices(self, query_embedding: List[float], top_k: int = 10) -> List[Device]:
        pass
