from typing import List
import requests
import numpy as np
from embeddings.base_embedding import ABaseEmbedding
from models.device import SmartHomeDevice
from models.embedded_device import EmbeddedDevice
from db_backends.base_db_backend import ABaseDbBackend

class OllamaEmbeddingModels:
    small = ("nomic-embed-text", 768) # 274 MB
    medium = "mxbai-embed-large" # 670 MB
    large = "BGE-M3" # 1.2 GB
    
class OllamaEmbedding(ABaseEmbedding):
    def __init__(self, ollama_url: str, db_backend: ABaseDbBackend, embedding_model = OllamaEmbeddingModels.small) -> None:
        self.ollama_url = ollama_url
        self.db_backend = db_backend
        self.embedding_model = embedding_model[0]
        self.embedding_length = embedding_model[1]

    def _extract_embedding(self, response: requests.Response) -> List[float]:
        data = response.json()
        if "embeddings" in data:
            return data["embeddings"][0]
        else:
            raise ValueError(f"Unexpected response: {data}")

    def embed_devices(self, devices: List[SmartHomeDevice]) -> None:      
        self.db_backend.cleanup_database(self.embedding_length)
          
        for device in devices:
            payload = {"model": self.embedding_model, "input": str(device)}
            response = requests.post(f"{self.ollama_url.removesuffix('/')}/api/embed", json=payload)
            response.raise_for_status()
            self.db_backend.save_device_embedding(EmbeddedDevice(device, self._extract_embedding(response)))        
    
    def embed_query(self, text: str) -> None:
        payload = {"model": self.embedding_model, "input": text}
        response = requests.post(f"{self.ollama_url.removesuffix('/')}/api/embed", json=payload)
        response.raise_for_status()
        
        return self._extract_embedding(response)