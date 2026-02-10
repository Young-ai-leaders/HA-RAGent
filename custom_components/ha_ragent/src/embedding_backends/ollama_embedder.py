from typing import Any, Dict, List
import logging
import aiohttp
import requests

from ..embedding_backends.base_embedder import ABaseEmbedder
from ..models.device import Device
from ..models.device_embedding import DeviceEmbedding

from homeassistant.core import HomeAssistant
from homeassistant.const import CONF_HOST, CONF_PORT, CONF_SSL
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from ..const import CONF_EMBEDDING_BACKEND_SECTION, CONF_LLM_BACKEND_SECTION

_logger = logging.getLogger(__name__)
    
class OllamaEmbedder(ABaseEmbedder):
    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]):
        super().__init__(hass, client_options)
    
    @staticmethod
    def get_name(client_options: Dict[str, Any]):
        return f"Embedding Backend: Ollama"

    @staticmethod
    async def async_validate_connection(hass: HomeAssistant, user_input: Dict[str, Any]) -> str | None:
        headers = {}
        try:
            session = async_get_clientsession(hass)
            response = await session.get(
                ABaseEmbedder._format_url(
                    hostname=user_input[CONF_EMBEDDING_BACKEND_SECTION][CONF_HOST],
                    port=user_input[CONF_EMBEDDING_BACKEND_SECTION][CONF_PORT],
                    ssl=user_input[CONF_EMBEDDING_BACKEND_SECTION][CONF_SSL],
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
        session = async_get_clientsession(self.hass)
        async with session.get(
             ABaseEmbedder._format_url(
                hostname=self.client_options[CONF_LLM_BACKEND_SECTION][CONF_HOST],
                port=self.client_options[CONF_LLM_BACKEND_SECTION][CONF_PORT],
                ssl=self.client_options[CONF_LLM_BACKEND_SECTION][CONF_SSL],
                path=f"/api/tags"
            ),
            timeout=aiohttp.ClientTimeout(total=5),
            headers=headers
        ) as response:
            response.raise_for_status()
            models_result = await response.json()

        return [x["name"] for x in models_result["models"] if "embed" in x["name"].lower()]

    def _extract_embedding(self, response: requests.Response) -> List[float]:
        data = response.json()
        if "embeddings" in data:
            return data["embeddings"][0]
        else:
            _logger.error(f"Unexpected response from embedding backend: {data}")

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