import asyncio
import time
from typing import Any, Dict, List
import logging
from pymongo import AsyncMongoClient, WriteConcern
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.asynchronous.collection import AsyncCollection

from homeassistant.core import HomeAssistant
from homeassistant.const import CONF_HOST, CONF_PORT, CONF_SSL

from .base_backend import ABaseDbBackend
from ...models.device import Device
from ...models.device_embedding import DeviceEmbedding

from ...const import (
    CONF_VECTOR_DB_NAME,
    CONF_VECTOR_DB_HOST,
    CONF_VECTOR_DB_PORT,
    CONF_VECTOR_DB_SSL,
    CONF_VECTOR_DB_USERNAME,
    CONF_VECTOR_DB_PASSWORD
)

_logger = logging.getLogger(__name__)

class MongoDbBackend(ABaseDbBackend):
    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]):
        super().__init__(hass, client_options)
        self.db_name = self.client_options.get(CONF_VECTOR_DB_NAME)
        self.url = MongoDbBackend._format_url(
            username=self.client_options.get(CONF_VECTOR_DB_USERNAME),
            password=self.client_options.get(CONF_VECTOR_DB_PASSWORD),
            hostname=self.client_options.get(CONF_VECTOR_DB_HOST),
            port=self.client_options.get(CONF_VECTOR_DB_PORT),
            ssl=self.client_options.get(CONF_VECTOR_DB_SSL),
        )
    
    @staticmethod
    def get_name(client_options: Dict[str, Any]):
        return f"DB Backend: MongoDB"

    @staticmethod
    def _format_url(username: str, password: str, hostname: str, port: str, ssl: bool) -> str:
        return f"mongodb://{username}:{password}@{hostname}:{port}/?ssl={'true' if ssl else 'false'}"
    
    def _get_connection(self) -> AsyncMongoClient:
        return AsyncMongoClient(self.url)
        
    def _get_database(self, connection: AsyncMongoClient) -> AsyncDatabase:
        return connection[self.db_name]
    
    def _get_collection(self, connection: AsyncMongoClient, collection_name: str) -> AsyncCollection:
        database = self._get_database(connection)
        return database[collection_name]

    async def _async_execute_and_verify(self, database: AsyncDatabase, command: Dict) -> bool:
        result = await database.command(command)
        return result.get("ok") == 1.0

    async def _async_database_exists(self, connection: AsyncMongoClient) -> bool:
        db_names = await connection.list_database_names()
        return self.db_name in db_names
    
    async def _async_collection_exists(self, connection: AsyncMongoClient, collection_name: str) -> bool:
        database = self._get_database(connection)
        collection_names = await database.list_collection_names()
        return collection_name in collection_names

    def _doc_to_device(self, doc: Dict[str, Any]) -> Device:
        return Device(
            id=doc.get("device_id"),
            name=doc.get("name"),
            device_type=doc.get("device_type"),
            area_name=doc.get("area_name"),
            capabilities=doc.get("capabilities", []),
            device_tags=doc.get("device_tags", []),
        )

    @staticmethod
    async def async_validate_connection(hass: HomeAssistant, user_input: Dict[str, Any]) -> str | None:
        conn = None
        try:
            url = MongoDbBackend._format_url(
                username=user_input.get(CONF_VECTOR_DB_USERNAME),
                password=user_input.get(CONF_VECTOR_DB_PASSWORD),
                hostname=user_input.get(CONF_VECTOR_DB_HOST),
                port=user_input.get(CONF_VECTOR_DB_PORT),
                ssl=user_input.get(CONF_VECTOR_DB_SSL),
            )
            connection = AsyncMongoClient(url)
            result = await connection.admin.command("ping")
            return None if result.get("ok") == 1.0 else "Failed to connect to MongoDB."
        except Exception as ex:
            return str(ex)
        finally:
            if conn:
                await conn.close()
    
    async def async_reset_database(self, config_subentry: dict, collection_name: str, embedding_length: int) -> None:
        conn = None
        try:                        
            conn = self._get_connection()
            database = self._get_database(conn)
            temp_collection_name = f"{collection_name}_temp"
            
            result = await self._async_execute_and_verify(database, {"create": temp_collection_name})
            if not result:
                _logger.warning(f"Collection {collection_name} creation failed.")
                return

            result = await self._async_execute_and_verify(database, {
                "createSearchIndexes": temp_collection_name,
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
            if not result:
                _logger.warning(f"Vector search index creation failed for collection {collection_name}")
                return
            
            wc = WriteConcern(w="majority", wtimeout=5000)
            result = await self._async_execute_and_verify(
                conn.admin, 
                {
                    "renameCollection": f"{self.db_name}.{temp_collection_name}",
                    "to": f"{self.db_name}.{collection_name}",
                    "dropTarget": True,
                    "writeConcern": wc.document
                }
            )
            if not result:
                _logger.warning(f"Renaming collection {temp_collection_name} to {collection_name} failed.")
                return
            
            _logger.info(f"Collection {collection_name} reset successfully")
        except Exception as e:
            _logger.error(f"Error resetting database: {e}", exc_info=True)
        finally:
            if conn:
                await conn.close()
            
    async def async_save_device_embeddings(self, config_subentry: dict, collection_name: str, device_embeddings: List[DeviceEmbedding]) -> None:
        conn = None
        try:
            conn = self._get_connection()
            collection = self._get_collection(conn, collection_name)
            for embedding in device_embeddings:
                await collection.insert_one(embedding.to_dict())
            _logger.info(f"Saved {len(device_embeddings)} device embeddings to collection {collection_name}")
        except Exception as e:
            _logger.error(f"Error saving device embeddings: {e}", exc_info=True)
        finally:
            if conn:
                await conn.close()
        
    async def async_retrieve_devices(self, config_subentry: dict, collection_name: str, query_embedding: List[float], top_k: int = 10) -> List[Device]:
        conn = None
        devices = []

        try:
            conn = self._get_connection()
            collection = self._get_collection(conn, collection_name)
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
                        "device_id": 1,
                        "name": 1,
                        "device_type": 1,
                        "area_name": 1,
                        "capabilities": 1,
                        "device_tags": 1
                    }
                }
            ]

            cursor = await collection.aggregate(pipeline)
            results = await cursor.to_list(length=top_k)
            
            devices = [self._doc_to_device(doc) for doc in results]
        except Exception as e:
            _logger.error(f"Error retrieving devices: {e}", exc_info=True)
        finally:
            if conn:
                await conn.close()
                
        return devices