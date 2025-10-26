from abc import ABC, abstractmethod
class BaseBackend(ABC):
    def __init__(self, template: str, model: str = "qwen3:8b"):
        self.template = template
        self.model = model
    
    @abstractmethod
    def send_request(self, query: str) -> str:
        pass