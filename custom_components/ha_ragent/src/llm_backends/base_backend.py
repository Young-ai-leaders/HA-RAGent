from abc import ABC, abstractmethod
from typing import Any, Dict

from homeassistant.core import HomeAssistant

class ALlmBaseBackend(ABC):
    def __init__(self, template: str, model: str):
        self.template = template
        self.model = model
    
    @abstractmethod
    def send_request(self, query: str) -> str:
        pass
    
    @staticmethod
    def get_name(client_options: dict[str, Any]):
        raise NotImplementedError()
    
    @staticmethod
    async def async_validate_connection(hass: HomeAssistant, user_input: Dict[str, Any]) -> str | None:
        raise NotImplementedError()