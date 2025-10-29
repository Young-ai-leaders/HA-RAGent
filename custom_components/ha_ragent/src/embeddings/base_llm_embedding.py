from typing import List
from abc import ABC, abstractmethod
from models.device import SmartHomeDevice

class ABaseEmbedding(ABC):
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def embed_text(self, text: str) -> None:
        pass
    
    @abstractmethod
    def embed_devices(self, devices: List[SmartHomeDevice]) -> None:
        pass