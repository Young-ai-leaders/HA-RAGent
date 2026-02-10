from typing import Any, Dict, List
import logging
from pymongo import MongoClient

from homeassistant.core import HomeAssistant
from homeassistant.const import CONF_HOST, CONF_PORT, CONF_SSL

from ..db_backends.base_db_backend import ABaseDbBackend
from ..models.device import Device
from ..models.device_embedding import DeviceEmbedding

from ..const import (
    CONF_VECTOR_DB_SECTION,
    CONF_VECTOR_DB_USERNAME,
    CONF_VECTOR_DB_PASSWORD
)

_logger = logging.getLogger(__name__)

class MongoDbBackend(ABaseDbBackend):
    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]):
        super().__init__(hass, client_options)
    
    @staticmethod
    def get_name(client_options: Dict[str, Any]):
        return f"DB Backend: MongoDB"

    @staticmethod
    def _format_url(username: str, password: str, hostname: str, port: str, ssl: bool) -> str:
        return f"mongodb://{username}:{password}@{hostname}:{port}/?ssl={'true' if ssl else 'false'}&directConnection=true"

    @staticmethod
    def _get_connection(url: str) -> MongoClient:
        return MongoClient(url)
        
    @staticmethod
    def _get_database(connection: MongoClient, db_name: str) -> None:
        return connection[db_name]
    
    @staticmethod
    def _get_collection(connection: MongoClient, db_name: str, collection_name: str) -> None:
        database = MongoDbBackend._get_database(connection, db_name)
        return database[collection_name]

    @staticmethod
    def _execute_and_verify(connection: MongoClient, command: Dict) -> None:
        result = MongoDbBackend._get_database(connection).command(command).get("ok")
        if result is None or result != 1.0:
            _logger.warning(f"Command execution failed with result: {result}")

        return result == 1.0

    @staticmethod
    async def async_validate_connection(hass: HomeAssistant, user_input: Dict[str, Any]) -> str | None:
        try:
            url = MongoDbBackend._format_url(
                username=user_input[CONF_VECTOR_DB_SECTION][CONF_VECTOR_DB_USERNAME],
                password=user_input[CONF_VECTOR_DB_SECTION][CONF_VECTOR_DB_PASSWORD],
                hostname=user_input[CONF_VECTOR_DB_SECTION][CONF_HOST],
                port=user_input[CONF_VECTOR_DB_SECTION][CONF_PORT],
                ssl=user_input[CONF_VECTOR_DB_SECTION][CONF_SSL],
            )
            connection = MongoDbBackend._get_connection(url)
            return None if connection.admin.command("ping")["ok"] == 1.0 else "Failed to connect to MongoDB"
        except Exception as ex:
            return str(ex)
    
    def cleanup_database(self, embedding_length: int) -> None:
        try:
            with self._get_connection() as conn:
                conn.drop_database(self.db_name)
            
                MongoDbBackend._execute_and_verify(conn, {"create": self.collection_name})
                MongoDbBackend._execute_and_verify(conn, {
                    "createSearchIndexes": self.collection_name,
                    "indexes": [
                        {
                            "name": "vector_search_index",
                            "type": "vectorSearch",
                            "definition": {
                                "fields": [
                                    {
                                        "path": "vector_embedding",
                                        "type": "vector",
                                        "numDimensions": embedding_length,
                                        "similarity": "cosine"
                                    }
                                ]
                            }
                        }
                    ]
                })
        except Exception as e:
            _logger.error(e)
            
    def save_device_embeddings(self, device_embeddings: List[DeviceEmbedding]) -> None:
        try:
            with self._get_connection() as conn:
                collection = self._get_collection(conn)
                for embedding in device_embeddings:
                    collection.insert_one(embedding.to_dict())
        except Exception as e:
            _logger.error(e)
        
    def retrieve_devices(self, query_embedding: List[float], top_k: int = 1) -> List[Device]:
        try:
            with self._get_connection() as conn:
                collection = self._get_collection(conn)
                pipeline = [
                    {
                        "$vectorSearch": {
                            "index": "vector_search_index",
                            "path": "vector_embedding",
                            "queryVector": query_embedding,
                            "numCandidates": top_k * 10,
                            "limit": top_k
                        }
                    },
                    {
                        "$project": {
                            "id": 1,
                            "name": 1,
                            "type": 1,
                            "location": 1,
                            "capabilities": 1,
                            "description": 1
                        }
                    }
                ]

                results = list(collection.aggregate(pipeline))
                            
            devices = []
            for doc in results:
                devices.append(Device(
                    id = doc.get("id"),
                    name = doc.get("name"),
                    type = doc.get("type"),
                    location = doc.get("location"),
                    capabilities = doc.get("capabilities", []),
                    description = doc.get("description", "")
                ))
            return devices
        except Exception as e:
            _logger.error(e)