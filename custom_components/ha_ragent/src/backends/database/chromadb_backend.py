import logging
import asyncio
import time
from typing import Any, Dict, List, Optional

from homeassistant.core import HomeAssistant

from chromadb import Client
from chromadb.config import Settings

from custom_components.ha_ragent.src.backends.database.base_backend import ABaseDbBackend

from ...const import (
    CONF_VECTOR_DB_HOST,
    CONF_VECTOR_DB_PORT,
    CONF_VECTOR_DB_SSL,
)

from ...models.device import Device
from ...models.device_embedding import DeviceEmbedding
from ...models.tool import LlmTool
from ...models.tool_embedding import LlmToolEmbedding

_logger = logging.getLogger(__name__)

class ChromaDbBackend(ABaseDbBackend):
    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]):
        super().__init__(hass, client_options)
        self._settings = Settings(
                chroma_api_impl="chromadb.api.fastapi.FastAPI",
                chroma_server_host=self.client_options.get(CONF_VECTOR_DB_HOST),
                chroma_server_http_port=self.client_options.get(CONF_VECTOR_DB_PORT),
                chroma_server_ssl_enabled=self.client_options.get(CONF_VECTOR_DB_SSL),
            )
            
        self._client = None

    @staticmethod
    def get_name(client_options: dict[str, Any]):
        return "DB: ChromaDB"

    @staticmethod
    def _validate_connection(client_options: dict[str, Any]) -> Optional[str]:
        host = client_options.get(CONF_VECTOR_DB_HOST)
        port = client_options.get(CONF_VECTOR_DB_PORT)
        ssl = client_options.get(CONF_VECTOR_DB_SSL)

        settings = Settings(
            chroma_api_impl="chromadb.api.fastapi.FastAPI",
            chroma_server_host=host,
            chroma_server_http_port=port,
            chroma_server_ssl_enabled=ssl,
        )
        client = Client(settings=settings)
        client.list_collections()
        return None

    @staticmethod
    async def async_validate_connection(hass: HomeAssistant, user_input: Dict[str, Any]) -> str | None:
        try:
            return await hass.async_add_executor_job(ChromaDbBackend._validate_connection, user_input)
        except Exception as e:
            _logger.error(f"Error validating ChromaDB connection: {e}", exc_info=True)

    def _get_client(self) -> Client:
        if self._client is None:
            self._client = Client(settings=self._settings)
        return self._client
    
    def _collection_exists(self, client: Client, collection_name: str) -> bool:
        collections = [col.name for col in client.list_collections()]
        return collection_name in collections
    
    def _save_device_embeddings(self, collection_name: str, device_embeddings: List[DeviceEmbedding]):
        collection = self._get_client().get_or_create_collection(name=collection_name)

        metadatas = []
        for emb in device_embeddings:
            meta = emb.to_dict()
            metadatas.append({k: v for k, v in meta.items() if not (isinstance(v, list) and len(v) == 0)})

        ids = [emb.device.id for emb in device_embeddings]
        embeddings = [emb.vector_embedding for emb in device_embeddings]
        collection.add(ids=ids, metadatas=metadatas, embeddings=embeddings)
        _logger.info(f"Saved {len(device_embeddings)} device embeddings to collection {collection_name}")

    def _query_devices(self, collection_name: str, query_embedding: List[float], top_k: int):
        collection = self._get_client().get_collection(name=collection_name)
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas"],
        )

    def _reset_collection(self, collection_name: str):
        try:
            if self._collection_exists(self._get_client(), collection_name):
                self._get_client().delete_collection(collection_name)
                time.sleep(1)

            self._get_client().create_collection(collection_name)
            _logger.info(f"Collection {collection_name} reset successfully")
        except Exception as e:
            _logger.error(f"Error resetting Chroma collection: {e}", exc_info=True)
    
    def _cleanup_database(self):
        try:
            for col in self._get_client().list_collections():
                self._get_client().delete_collection(col.name)
            _logger.info(f"Database cleanup for {self._get_client()} successful.")
        except Exception as e:
             _logger.error(f"Error cleaning up database: {e}", exc_info=True)

    async def async_cleanup_database(self) -> None:
        try:
            await self.hass.async_add_executor_job(self._cleanup_database)
        except Exception as e:
             _logger.error(f"Error cleaning up database: {e}", exc_info=True)

    async def async_reset_database(self, config_subentry: dict, collection_name: str, embedding_length: int) -> None:
        try:
            await self.hass.async_add_executor_job(self._reset_collection, collection_name)
        except Exception as e:
            _logger.error(f"Error resetting database: {e}", exc_info=True)

    async def async_save_object_embeddings(self, config_subentry: dict, collection_name: str, device_embeddings: List[DeviceEmbedding | LlmToolEmbedding]) -> None:
        try:
            await self.hass.async_add_executor_job(self._save_device_embeddings, collection_name, device_embeddings)
        except Exception as e:
             _logger.error(f"Error saving device embeddings: {e}", exc_info=True)

    async def async_retrieve_objects(self, config_subentry: dict, collection_name: str, query_embedding: List[float], top_k: int = 10) -> List[Device | LlmTool]:
        devices: List[Device] = []
        try:
            result = await self.hass.async_add_executor_job(self._query_devices, collection_name, query_embedding, top_k)
            metadata = result.get("metadatas") or []
            if metadata:
                devices = [DeviceEmbedding.from_dict(m) for m in metadata[0]]
        except Exception as e:
            _logger.error(f"Error retrieving devices: {e}", exc_info=True)
        return devices

