from typing import Any, Dict, List
import logging
import aiohttp

from .base_backend import ABaseEmbedder
from ...models.device import Device
from ...models.device_embedding import DeviceEmbedding

from homeassistant.core import HomeAssistant
from homeassistant.const import CONF_HOST, CONF_PORT, CONF_SSL
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from ...const import (
    CONF_EMBEDDING_BACKEND_SECTION, 
    CONF_EMBEDDING_MODEL
)

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
                hostname=self.client_options[CONF_EMBEDDING_BACKEND_SECTION][CONF_HOST],
                port=self.client_options[CONF_EMBEDDING_BACKEND_SECTION][CONF_PORT],
                ssl=self.client_options[CONF_EMBEDDING_BACKEND_SECTION][CONF_SSL],
                path=f"/api/tags"
            ),
            timeout=aiohttp.ClientTimeout(total=5),
            headers=headers
        ) as response:
            response.raise_for_status()
            models_result = await response.json()

        return [x["name"] for x in models_result["models"] if "embed" in x["name"].lower()]

    async def async_embed_text(self, config_subentry: dict, text: str) -> List[float]:
        try:
            session = async_get_clientsession(self.hass)
            url = ABaseEmbedder._format_url(
                hostname=self.client_options[CONF_EMBEDDING_BACKEND_SECTION][CONF_HOST],
                port=self.client_options[CONF_EMBEDDING_BACKEND_SECTION][CONF_PORT],
                ssl=self.client_options[CONF_EMBEDDING_BACKEND_SECTION][CONF_SSL],
                path="/api/embed"
            )
            
            payload = {
                "model": config_subentry[CONF_EMBEDDING_MODEL],
                "input": text
            }
            
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
                response.raise_for_status()
                data = await response.json()
                if "embeddings" in data and len(data["embeddings"]) > 0:
                    return data["embeddings"][0]
                else:
                    _logger.error(f"Unexpected response from embedding backend: {data}")
                    return []
        except Exception as e:
            _logger.error(f"Error embedding text: {e}", exc_info=True)
            return []

    async def async_embed_devices(self, config_subentry: dict, devices: List[Device]) -> List[DeviceEmbedding]:
        return [DeviceEmbedding(device=device, vector_embedding=await self.async_embed_text(config_subentry, str(device))) for device in devices]