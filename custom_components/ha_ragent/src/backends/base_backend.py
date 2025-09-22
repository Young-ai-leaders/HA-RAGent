from abc import ABC, abstractmethod
class BaseBackend(ABC):
    def __init__(self):
        pass

    def send_user_query(self, query: str) -> str:
        pass