from typing import List
import logging
import requests

from ..embeddings.base_embedder import ABaseEmbedder
from ..models.device import Device
from ..models.device_embedding import DeviceEmbedding
from ..models.embedding_model import EmbeddingModel
from ..db_backends.base_db_backend import ABaseDbBackend

_logger = logging.getLogger(__name__)

class OllamaEmbeddingModels:
    small = EmbeddingModel("nomic-embed-text", 768) # 274 MB
    medium = EmbeddingModel("mxbai-embed-large", 768) # 670 MB
    large = EmbeddingModel("BGE-M3", 768) # 1.2 GB
    
class OllamaEmbedder(ABaseEmbedder):
    def __init__(self, ollama_url: str, db_backend: ABaseDbBackend, model: EmbeddingModel = OllamaEmbeddingModels.small) -> None:
        self.ollama_url = ollama_url
        self.db_backend = db_backend
        self.model = model
        
    def get_embedding_size(self) -> int:
        embedding = self.embed_text("This is a test message")
        return len(embedding)

    def _extract_embedding(self, response: requests.Response) -> List[float]:
        data = response.json()
        if "embeddings" in data:
            return data["embeddings"][0]
        else:
            _logger.error(f"Unexpected response: {data}")

    def embed_text(self, text: str) -> None:
        try:
            payload = {"model": self.model.model_name, "input": text}
            response = requests.post(f"{self.ollama_url.removesuffix('/')}/api/embed", json=payload)
            response.raise_for_status()
            return self._extract_embedding(response)
        except Exception as e:
            _logger.error(e)

    def embed_devices(self, devices: List[Device]) -> None: 
        try:
            self.db_backend.cleanup_database(self.model.embedding_size)
            embedded_devices = [DeviceEmbedding(device, self.embed_text(str(device))) for device in devices]                
            self.db_backend.save_device_embeddings(embedded_devices)
        except Exception as e:
            _logger.error(e)