import logging
import asyncio
import os
from typing import Any, Dict, List, Optional
from uuid import uuid4

from homeassistant.core import HomeAssistant

import faiss
import numpy as np
import pickle

from custom_components.ha_ragent.src.backends.database.base_backend import ABaseDbBackend

from ...const import (
    CONF_VECTOR_DB_NAME
)

from ...models.device import Device
from ...models.device_embedding import DeviceEmbedding
from ...models.tool import LlmTool
from ...models.tool_embedding import LlmToolEmbedding

_logger = logging.getLogger(__name__)

class FaissDbBackend(ABaseDbBackend):
    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]):
        super().__init__(hass, client_options)
        self._storage_path = hass.config.path("ha_ragent_storage")
        self.db_name = self.client_options.get(CONF_VECTOR_DB_NAME)

        os.makedirs(os.path.join(self._storage_path, self.db_name), exist_ok=True)
        
        self._indices: Dict[str, faiss.Index] = {}
        self._metadata: Dict[str, List[Dict[str, Any]]] = {}

    @staticmethod
    def get_name(client_options: dict[str, Any]):
        return "DB: Local FAISS"

    @staticmethod
    async def async_validate_connection(hass: HomeAssistant, user_input: Dict[str, Any]) -> str | None:
        return None
    
    def _get_paths(self, collection_name: str):
        index_path = os.path.join(self._storage_path, self.db_name, f"{collection_name}.index")
        meta_path = os.path.join(self._storage_path, self.db_name, f"{collection_name}.pkl")
        return index_path, meta_path

    def _load_collection(self, collection_name: str, embedding_length: int = 1536):
        """Lazy load or initialize the index and metadata."""
        idx_path, meta_path = self._get_paths(collection_name)
        
        if collection_name not in self._indices:
            if os.path.exists(idx_path) and os.path.exists(meta_path):
                try:
                    self._indices[collection_name] = faiss.read_index(idx_path)
                    with open(meta_path, "rb") as f:
                        self._metadata[collection_name] = pickle.load(f)
                except Exception as e:
                    _logger.error(f"Failed to load collection {collection_name}: {e}")
                    self._create_empty(collection_name, embedding_length)
            else:
                self._create_empty(collection_name, embedding_length)

    def _create_empty(self, collection_name: str, embedding_length: int):
        self._indices[collection_name] = faiss.IndexFlatL2(embedding_length)
        self._metadata[collection_name] = []

    def _save_to_disk(self, collection_name: str):
        idx_path, meta_path = self._get_paths(collection_name)
        faiss.write_index(self._indices[collection_name], idx_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self._metadata[collection_name], f)

    def _save_device_embeddings(self, collection_name: str, device_embeddings: List[DeviceEmbedding | LlmToolEmbedding]):
        if not device_embeddings:
            return

        dim = len(device_embeddings[0].vector_embedding)
        self._load_collection(collection_name, dim)

        vectors = np.array([emb.vector_embedding for emb in device_embeddings]).astype('float32')
        metadatas = [emb.to_dict() for emb in device_embeddings]

        self._indices[collection_name].add(vectors)
        self._metadata[collection_name].extend(metadatas)
        
        self._save_to_disk(collection_name)
        _logger.info(f"Saved {len(device_embeddings)} embeddings to local FAISS index: {collection_name}")

    def _query_devices(self, collection_name: str, query_embedding: List[float], top_k: int):
        self._load_collection(collection_name, len(query_embedding))
        
        query_vector = np.array([query_embedding]).astype('float32')
        distances, indices = self._indices[collection_name].search(query_vector, top_k)

        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self._metadata[collection_name]):
                results.append(self._metadata[collection_name][idx])
        return results
    
    def _cleanup_database(self):
        for filename in os.listdir(self._storage_path):
            if filename.endswith((".index", ".pkl")):
                os.remove(os.path.join(self._storage_path, filename))
        self._indices.clear()
        self._metadata.clear()
        
    def _reset_database(self, collection_name: str, embedding_length: int):
        idx_path, meta_path = self._get_paths(collection_name)

        self._indices.pop(collection_name)
        self._metadata.pop(collection_name)

        for path in (idx_path, meta_path):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as err:
                    _logger.warning(f"Failed to remove stale FAISS file {path}: {err}")

        self._create_empty(collection_name, embedding_length)
        self._save_to_disk(collection_name)

    async def async_cleanup_database(self) -> None:
        try:
            await self.hass.async_add_executor_job(self._cleanup_database)
        except Exception as e:
             _logger.error(f"Error cleaning up database: {e}", exc_info=True)

    async def async_reset_database(self, config_subentry: dict, collection_name: str, embedding_length: int) -> None:
        try:
            await self.hass.async_add_executor_job(self._reset_database, collection_name, embedding_length)
        except Exception as e:
            _logger.error(f"Error resetting database: {e}", exc_info=True)

    async def async_save_object_embeddings(self, config_subentry: dict, collection_name: str, device_embeddings: List[DeviceEmbedding | LlmToolEmbedding]) -> None:
        try:
            await self.hass.async_add_executor_job(self._save_device_embeddings, collection_name, device_embeddings)
        except Exception as e:
             _logger.error(f"Error saving device embeddings: {e}", exc_info=True)

    async def async_retrieve_objects(self, object_type: type[DeviceEmbedding | LlmToolEmbedding], config_subentry: dict, collection_name: str, query_embedding: List[float], top_k: int = 10) -> List[Device | LlmTool]:
        devices: List[Device] = []
        try:
            results = await self.hass.async_add_executor_job(self._query_devices, collection_name, query_embedding, top_k)
            devices = [object_type.parse_object(m) for m in results]
        except Exception as e:
            _logger.error(f"Error retrieving devices: {e}", exc_info=True)
        return devices

