from typing import List
from abc import ABC, abstractmethod
from models.device import SmartHomeDevice
from models.device_embedding import DeviceEmbedding

class ABaseDbBackend(ABC):
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def cleanup_database(self, embedding_length: int) -> None:
        pass
    
    @abstractmethod
    def save_device_embeddings(self, device_embeddings: List[DeviceEmbedding]) -> None:
        pass
    
    @abstractmethod
    def retrieve_devices(self, query_embedding: List[float], top_k: int = 10) -> List[SmartHomeDevice]:
        pass
