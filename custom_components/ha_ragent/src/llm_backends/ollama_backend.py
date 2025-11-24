import aiohttp
import logging
from typing import Any, Dict, List

from homeassistant.core import HomeAssistant
from homeassistant.const import CONF_HOST, CONF_PORT, CONF_SSL
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from ..llm_backends.base_backend import ALlmBaseBackend
from ..utils import format_url

_logger = logging.getLogger(__name__)

class OllamaBackend(ALlmBaseBackend):
    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]):
        super().__init__(hass, client_options)

    def send_request(self, query):
        return super().send_request(query)
    
    @staticmethod
    def get_name(client_options: Dict[str, Any]):
        host = client_options[CONF_HOST]
        port = client_options[CONF_PORT]
        ssl = client_options[CONF_SSL]
        path = "/"
        return f"Ollama at '{format_url(hostname=host, port=port, ssl=ssl, path=path)}'"
    
    @staticmethod
    async def async_validate_connection(hass: HomeAssistant, user_input: Dict[str, Any]) -> str | None:
        headers = {}
        try:
            session = async_get_clientsession(hass)
            response = await session.get(
                format_url(
                    hostname=user_input[CONF_HOST],
                    port=user_input[CONF_PORT],
                    ssl=user_input[CONF_SSL],
                    path=f"/api/tags"
                ),
                timeout=aiohttp.ClientTimeout(total=5),
                headers=headers
            )
            return None if response.ok else f"HTTP Status {response.status}"
        except Exception as ex:
            return str(ex)
    
    async def async_get_available_models(self) -> List[str]:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        session = async_get_clientsession(self.hass)
        async with session.get(
            f"{self.api_host}/api/tags",
            timeout=aiohttp.ClientTimeout(total=5),
            headers=headers
        ) as response:
            response.raise_for_status()
            models_result = await response.json()

        return [x["name"] for x in models_result["models"]]