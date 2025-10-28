from typing import List
from abc import ABC, abstractmethod

class ABaseDbBackend(ABC):
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def save_device_embedding(self, embedding: str) -> None:
        pass
    
    @abstractmethod
    def cleanup_database(self, embedding_length: int) -> None:
        pass
