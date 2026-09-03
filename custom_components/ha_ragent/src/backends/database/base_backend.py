from typing import Any, Dict, List
from abc import ABC, abstractmethod

try:
    from homeassistant.core import HomeAssistant
except ImportError:
    from custom_components.ha_ragent.src.mock import MockHomeAssistant as HomeAssistant

from custom_components.ha_ragent.src.models.device import Device
from custom_components.ha_ragent.src.models.device_embedding import DeviceEmbedding
from custom_components.ha_ragent.src.models.tool import LlmTool
from custom_components.ha_ragent.src.models.tool_embedding import LlmToolEmbedding
from custom_components.ha_ragent.src.models.memory import Memory
from custom_components.ha_ragent.src.models.memory_embedding import MemoryEmbedding

class ABaseDbBackend(ABC):
    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]):
        self.hass = hass
        self.client_options = client_options

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """Return the name of the database backend."""
        return "DB"
    
    @staticmethod
    @abstractmethod
    async def async_validate_connection(hass: HomeAssistant, user_input: Dict[str, Any]) -> str | None:
        """Validate the connection to the database backend."""
        raise NotImplementedError()

    @abstractmethod
    async def async_ensure_collection_exists(self, config_subentry: dict, collection_name: str, embedding_length: int) -> None:
        """Create a collection without removing existing objects."""
        raise NotImplementedError()

    @abstractmethod
    async def async_cleanup_database(self) -> None:
        """Cleanup the database, removing all collections and data."""
        raise NotImplementedError()

    @abstractmethod
    async def async_reset_collection(self, config_subentry: dict, collection_name: str, embedding_length: int) -> None:
        """Delete and recreate a collection."""
        raise NotImplementedError()

    @abstractmethod
    async def async_cleanup_collection(self, config_subentry: dict, collection_name: str) -> None:
        """Delete all objects in a collection."""
        raise NotImplementedError()

    @abstractmethod
    async def async_upsert_objects(self, config_subentry: dict, collection_name: str, id_field: str, object_embeddings: List[MemoryEmbedding]) -> None:
        """Insert objects or replace records with the same id."""
        raise NotImplementedError()

    @abstractmethod
    async def async_save_objects(self, config_subentry: dict, collection_name: str, device_embeddings: List[DeviceEmbedding | LlmToolEmbedding | MemoryEmbedding]) -> None:
        """Insert objects without replacing existing records."""
        raise NotImplementedError()

    @abstractmethod
    async def async_retrieve_objects(self, object_type: type[DeviceEmbedding | LlmToolEmbedding | MemoryEmbedding], config_subentry: dict, collection_name: str, query_embedding: List[float], top_k: int = 10) -> List[Device | LlmTool | Memory]:
        """Retrieve objects from the database based on a query embedding."""
        raise NotImplementedError()

    @abstractmethod
    async def async_list_objects(self, object_type: type[DeviceEmbedding | LlmToolEmbedding | MemoryEmbedding], config_subentry: dict, collection_name: str) -> List[Device | LlmTool | Memory]:
        """List all objects in a collection."""
        raise NotImplementedError()

    async def async_increment_memory_retrieval_counts(self, config_subentry: dict, collection_name: str, memory_ids: List[str]) -> None:
        """Increment retrieval counts for memory records."""
        raise NotImplementedError()

    @abstractmethod
    async def async_delete_objects(self, config_subentry: dict, collection_name: str, id_field: str, object_ids: List[str]) -> int:
        """Delete objects by id."""
        raise NotImplementedError()
