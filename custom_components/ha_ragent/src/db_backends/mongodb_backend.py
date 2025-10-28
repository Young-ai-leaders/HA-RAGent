from typing import List
from pymongo import MongoClient
from db_backends.base_db_backend import ABaseDbBackend
from models.device import SmartHomeDevice
from models.embedded_device import EmbeddedDevice

class MongoDbBackend(ABaseDbBackend):
    def __init__(self, host: str, username: str, password: str, port: str = "27017", db_name: str = "homeassistant", collection_name: str = "device_embeddings") -> None:
        self.mongo_uri = f"mongodb://{username}:{password}@{host}:{port}/?directConnection=true"
        self.db_name = db_name
        self.collection_name = collection_name
        self.connection = None
    
    def _setup_connection(self) -> None:
        if self.connection != None:
            return
        
        try:
            self.connection = MongoClient(self.mongo_uri)
            self.connection.admin.command("ping")
        except Exception as e:
            self.connection = None
            raise ConnectionError(f"Failed to connect to MongoDB: {e}")
    
    def _get_database(self):
        self._setup_connection()
        return self.connection[self.db_name]
    
    def _get_collection(self):
        db = self._get_database()
        return db[self.collection_name]
    
    def cleanup_database(self, embedding_length: int):
        self._setup_connection()
        self.connection.drop_database(self.db_name)
        
        collection = self._get_collection()
        collection.insert_one({"init": True})
        
        db = self._get_database()
        result = db.command({
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
                                "numDimensions": embedding_length,  # Adjust to your vector size
                                "similarity": "cosine"  # or "dotProduct" or "euclidean"
                            }
                        ]
                    }
                }
            ]
        })
        
        if result.get("ok") != 1.0:
            raise f"Vector index creation returned: {result}"
    
    def save_device_embedding(self, embedding: EmbeddedDevice) -> None:
        collection = self._get_collection()
        collection.insert_one(embedding.to_dict())
        
    def get_top_n_devices_embeddings(self, query_embedding, top_k = 1) -> List[SmartHomeDevice]:
        collection = self._get_collection()
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

        try:
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