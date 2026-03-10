import logging
import asyncio
from typing import Any, Dict, List, Optional

from homeassistant.core import HomeAssistant
from chromadb import Client

from custom_components.ha_ragent.src.backends.database.base_backend import ABaseDbBackend

from ...models.device import Device
from ...models.device_embedding import DeviceEmbedding

_logger = logging.getLogger(__name__)

def _run_in_executor(hass: Optional[HomeAssistant], func, *args, **kwargs):
    if hass:
        return hass.async_add_executor_job(func, *args, **kwargs)

    loop = asyncio.get_running_loop()
    return loop.run_in_executor(None, lambda: func(*args, **kwargs))


class ChromaDbBackend(ABaseDbBackend):
    """Vector database backend using a Chroma server or in‑memory instance."""
    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]):
        super().__init__(hass, client_options)
        self._client = None

    @staticmethod
    def get_name(client_options: dict[str, Any]):
        return "DB Backend: ChromaDB"

    @staticmethod
    async def async_validate_connection(hass: HomeAssistant, user_input: Dict[str, Any]) -> str | None:
        try:
            client = Client()
            client.list_collections()
            return None
        except Exception as e:
            return str(e)

    def _get_client(self):
        if self._client is None:
            self._client = Client()
            
        return self._client
    
    def _doc_to_device(self, doc: Dict[str, Any]) -> Device:
        return Device(
            id=doc.get("device_id"),
            name=doc.get("name"),
            domain=doc.get("domain"),
            area_name=doc.get("area_name"),
            device_tags=doc.get("device_tags", []),
            services=doc.get("services", [])
        )

    async def async_cleanup_database(self) -> None:
        try:
            client = self._get_client()
            for col in client.list_collections():
                client.delete_collection(col.name)
            _logger.info("Chroma: cleaned up all collections")
        except Exception as exc:
            _logger.error("Error cleaning up Chroma database: %s", exc, exc_info=True)

    async def async_reset_database(self, config_subentry: dict, collection_name: str, embedding_length: int) -> None:
        try:
            def _reset():
                client = self._get_client()
                existing = [c.name for c in client.list_collections()]
                if collection_name in existing:
                    client.delete_collection(collection_name)
                client.create_collection(collection_name)

            await _run_in_executor(self.hass, _reset)
            _logger.info("Chroma collection %s reset successfully", collection_name)
        except Exception as exc:
            _logger.error("Error resetting Chroma collection: %s", exc, exc_info=True)

    async def async_save_device_embeddings(self, config_subentry: dict, collection_name: str, device_embeddings: List[DeviceEmbedding]) -> None:
        try:
            def _save():
                client = self._get_client()
                collection = client.get_or_create_collection(name=collection_name)
                for emb in device_embeddings:
                    collection.add(
                        ids=[emb.device.id],
                        metadatas=[emb.to_dict()],
                        embeddings=[emb.vector_embedding],
                    )

            await _run_in_executor(self.hass, _save)
            _logger.info("Saved %d device embeddings to Chroma collection %s", len(device_embeddings), collection_name)
        except Exception as exc:
            _logger.error("Error saving device embeddings to Chroma: %s", exc, exc_info=True)

    async def async_retrieve_devices(self, config_subentry: dict, collection_name: str, query_embedding: List[float], top_k: int = 10) -> List[Device]:
        devices: List[Device] = []
        try:
            def _query():
                client = self._get_client()
                collection = client.get_collection(name=collection_name)
                return collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["metadatas"],
                )

            result = await _run_in_executor(self.hass, _query)
            metadatas = result.get("metadatas") or []
            if metadatas:
                devices = [self._doc_to_device(m) for m in metadatas[0]]
        except Exception as exc:
            _logger.error("Error retrieving devices from Chroma: %s", exc, exc_info=True)
        return devices

