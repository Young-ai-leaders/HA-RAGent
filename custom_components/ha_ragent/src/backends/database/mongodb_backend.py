import asyncio
import time
from typing import Any, Dict, List
import logging
from pymongo import AsyncMongoClient, WriteConcern
from pymongo.errors import OperationFailure
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.asynchronous.collection import AsyncCollection

from homeassistant.core import HomeAssistant
from homeassistant.const import CONF_HOST, CONF_PORT, CONF_SSL

from .base_backend import ABaseDbBackend
from ...models.device import Device
from ...models.device_embedding import DeviceEmbedding
from ...models.tool import LlmTool
from ...models.tool_embedding import LlmToolEmbedding

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
        return f"DB: MongoDB"

    @staticmethod
    def _format_url(username: str, password: str, hostname: str, port: str, ssl: bool) -> str:
        return f"mongodb://{username}:{password}@{hostname}:{port}/?ssl={'true' if ssl else 'false'}&directConnection=true"
    
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

    async def _async_vector_index_exists(self, database: AsyncDatabase, collection_name: str, index_name: str) -> bool:
        try:
            result = await database.command({
                "listSearchIndexes": collection_name,
                "name": index_name,
            })
            indexes = result.get("indexes", [])
            return any(index.get("name") == index_name for index in indexes)
        except OperationFailure as err:
            if err.code == 125:
                _logger.warning(
                    "Search Index Management service unavailable while listing indexes for %s; "
                    "skipping vector index initialization for now.",
                    collection_name,
                )
                return True
            raise
    
    async def _async_init_database(self, conn: AsyncMongoClient, database: AsyncDatabase, collection_name: str, embedding_length: int) -> None:
        if not await self._async_collection_exists(conn, collection_name):
            await database.create_collection(collection_name)

        if not await self._async_vector_index_exists(database, collection_name, "vector_search_index"):
            try:
                result = await self._async_execute_and_verify(database, {
                    "createSearchIndexes": collection_name,
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
            except OperationFailure as err:
                if err.code == 125:
                    _logger.warning(
                        "Search Index Management service unavailable while creating vector index for %s; "
                        "continuing without resetting the index.",
                        collection_name,
                    )
                else:
                    raise

    @staticmethod
    async def async_validate_connection(hass: HomeAssistant, user_input: Dict[str, Any]) -> str | None:
        connection = None
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
            if connection:
                await connection.close()

    async def async_cleanup_database(self) -> None:
        conn = None
        try:
            conn = self._get_connection()
            await conn.drop_database(self.db_name)
            _logger.info(f"Database cleanup for {self.db_name} successful.")
        except Exception as e:
            _logger.error(f"Error cleaning up database: {e}", exc_info=True)
        finally:
            if conn:
                await conn.close()
    
    async def async_reset_database(self, config_subentry: dict, collection_name: str, embedding_length: int) -> None:
        conn = None
        try:
            conn = self._get_connection()
            database = self._get_database(conn)

            await self._async_init_database(conn, database, collection_name, embedding_length)
            await database[collection_name].delete_many({})

            _logger.info(f"Collection {collection_name} reset successfully")
        except Exception as e:
            _logger.error(f"Error resetting database: {e}", exc_info=True)
        finally:
            if conn:
                await conn.close()
            
    async def async_save_object_embeddings(self, config_subentry: dict, collection_name: str, device_embeddings: List[DeviceEmbedding | LlmToolEmbedding]) -> None:
        conn = None
        try:
            conn = self._get_connection()
            collection = self._get_collection(conn, collection_name)
            await collection.insert_many([embedding.to_dict() for embedding in device_embeddings], ordered=False)
            _logger.info(f"Saved {len(device_embeddings)} device embeddings to collection {collection_name}")
        except Exception as e:
            _logger.error(f"Error saving device embeddings: {e}", exc_info=True)
        finally:
            if conn:
                await conn.close()
        
    async def async_retrieve_objects(self, object_type: type[DeviceEmbedding | LlmToolEmbedding], config_subentry: dict, collection_name: str, query_embedding: List[float], top_k: int = 10) -> List[Device | LlmTool]:
        conn = None
        devices = []

        try:
            conn = self._get_connection()
            collection = self._get_collection(conn, collection_name)

            if object_type == DeviceEmbedding:
                projection = {
                    "device_id": 1,
                    "name": 1,
                    "domain": 1,
                    "area_name": 1,
                    "device_labels": 1,
                    "services": 1,
                    "aliases": 1
                }
            elif object_type == LlmToolEmbedding:
                projection = {
                    "name": 1,
                    "description": 1,
                    "parameters": 1,
                    "metadata": 1
                }
            else:
                _logger.error(f"Unsupported object type for retrieval: {object_type}")
                return []

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
                    "$project": projection
                }
            ]

            cursor = await collection.aggregate(pipeline)
            results = await cursor.to_list(length=top_k)
            
            devices = [object_type.parse_object(doc) for doc in results]
        except Exception as e:
            _logger.error(f"Error retrieving devices: {e}", exc_info=True)
        finally:
            if conn:
                await conn.close()
                
        return devices