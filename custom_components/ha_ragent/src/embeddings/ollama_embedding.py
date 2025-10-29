from typing import List
import logging
import requests
from embeddings.base_llm_embedding import ABaseEmbedding
from models.device import SmartHomeDevice
from models.device_embedding import DeviceEmbedding
from db_backends.base_db_backend import ABaseDbBackend

_logger = logging.getLogger(__name__)

class OllamaEmbeddingModels:
    small = ("nomic-embed-text", 768) # 274 MB
    medium = ("mxbai-embed-large", 768) # 670 MB
    large = ("BGE-M3", 768) # 1.2 GB
    
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

    def embed_text(self, text: str) -> None:
        try:
            payload = {"model": self.embedding_model, "input": text}
            response = requests.post(f"{self.ollama_url.removesuffix('/')}/api/embed", json=payload)
            response.raise_for_status()
            return self._extract_embedding(response)
        except Exception as e:
            _logger.error(e)

    def embed_devices(self, devices: List[SmartHomeDevice]) -> None: 
        try:
            self.db_backend.cleanup_database(self.embedding_length)
            embedded_devices = [DeviceEmbedding(device, self.embed_text(str(device))) for device in devices]                
            self.db_backend.save_device_embeddings(embedded_devices)
        except Exception as e:
            _logger.error(e)