from typing import List, Dict
from pymongo import MongoClient
from db_backends.base_db_backend import ABaseDbBackend
from models.device import SmartHomeDevice
from models.device_embedding import DeviceEmbedding

class MongoDbBackend(ABaseDbBackend):
    def __init__(self, host: str, username: str, password: str, port: str = "27017", db_name: str = "homeassistant", collection_name: str = "device_embeddings") -> None:
        self.mongo_uri = f"mongodb://{username}:{password}@{host}:{port}/?directConnection=true"
        self.db_name = db_name
        self.collection_name = collection_name
    
    def _get_connection(self) -> None:
        return MongoClient(self.mongo_uri)
        
    def _get_database(self, connection: MongoClient) -> None:
        return connection[self.db_name]
    
    def _get_collection(self, connection: MongoClient) -> None:
        database = self._get_database(connection)
        return database[self.collection_name]
    
    def _execute_and_verify(self, connection: MongoClient, command: Dict) -> None:
        result = self._get_database(connection).command(command)
        if result.get("ok") != 1.0:
            raise f"Command execution failed with result: {result}"
        
    def cleanup_database(self, embedding_length: int) -> None:
        with self._get_connection() as conn:
            conn.drop_database(self.db_name)
        
            self._execute_and_verify(conn, {"create": self.collection_name})
            self._execute_and_verify(conn, {
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
            
    def save_device_embeddings(self, device_embeddings: List[DeviceEmbedding]) -> None:
        with self._get_connection() as conn:
            collection = self._get_collection(conn)
            for embedding in device_embeddings:
                collection.insert_one(embedding.to_dict())
        
    def load_device_embeddings(self, query_embedding: List[float], top_k: int = 1) -> List[SmartHomeDevice]:
        with self._get_connection() as conn:
            try:
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
            except Exception as e:
                raise RuntimeError(f"Error during vector search: {e}")
                            
            devices = []
            for doc in results:
                devices.append(SmartHomeDevice(
                    id = doc.get("id"),
                    name = doc.get("name"),
                    type = doc.get("type"),
                    location = doc.get("location"),
                    capabilities = doc.get("capabilities", []),
                    description = doc.get("description", "")
                ))
            return devices