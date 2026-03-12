import asyncio
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
    CONF_EMBEDDING_MODEL,
    CONF_EMBEDDING_HOST,
    CONF_EMBEDDING_PORT,
    CONF_EMBEDDING_SSL
)

_logger = logging.getLogger(__name__)
    
class OllamaEmbedder(ABaseEmbedder):
    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]):
        super().__init__(hass, client_options)
    
    @staticmethod
    def get_name(client_options: Dict[str, Any]):
        return f"Embedder: Ollama"

    @staticmethod
    async def async_validate_connection(hass: HomeAssistant, user_input: Dict[str, Any]) -> str | None:
        headers = {}
        try:
            session = async_get_clientsession(hass)
            response = await session.get(
                ABaseEmbedder._format_url(
                    hostname=user_input.get(CONF_EMBEDDING_HOST),
                    port=user_input.get(CONF_EMBEDDING_PORT),
                    ssl=user_input.get(CONF_EMBEDDING_SSL),
                    path=f"/api/tags"
                ),
                timeout=aiohttp.ClientTimeout(total=5),
                headers=headers
            )
            return None if response.ok else f"HTTP Status {response.status}"
        except Exception as ex:
            return str(ex)
    
    async def _async_get_model_info(self, model_name: str) -> Dict[str, Any]:
        session = async_get_clientsession(self.hass)
        async with session.post(
            ABaseEmbedder._format_url(
                hostname=self.client_options.get(CONF_EMBEDDING_HOST),
                port=self.client_options.get(CONF_EMBEDDING_PORT),
                ssl=self.client_options.get(CONF_EMBEDDING_SSL),
                path="/api/show",
            ),
            json={"model": model_name},
            timeout=aiohttp.ClientTimeout(total=5),
            headers={},
        ) as response:
            response.raise_for_status()
            model_result = await response.json()

        capabilities = model_result.get("capabilities", [])
        is_tool = "tools" in capabilities
        is_embedding = "embedding" in capabilities

        return {
            "name": model_name,
            "supports_tools": is_tool,
            "is_embedding": is_embedding
        }

    async def async_preload_model(self, config_subentry: dict) -> None:
        await self.async_embed_text(config_subentry, "Preloading model with a test embedding request.", keep_alive=-1)  
    
    async def async_unload_model(self, config_subentry: dict) -> None:
        await self.async_embed_text(config_subentry, "Unloading model with a test embedding request.", keep_alive=0)
    
    async def async_get_available_models(self) -> List[str]:
        session = async_get_clientsession(self.hass)
        async with session.get(
             ABaseEmbedder._format_url(
                hostname=self.client_options.get(CONF_EMBEDDING_HOST),
                port=self.client_options.get(CONF_EMBEDDING_PORT),
                ssl=self.client_options.get(CONF_EMBEDDING_SSL),
                path=f"/api/tags"
            ),
            timeout=aiohttp.ClientTimeout(total=5),
            headers={}
        ) as response:
            response.raise_for_status()
            models_result = await response.json()

        names = [x["name"] for x in models_result.get("models", [])]
        infos = await asyncio.gather(*(self._async_get_model_info(name) for name in names), return_exceptions=True)
        available = []
        for info in infos:
            if isinstance(info, Exception):
                continue
            if info.get("is_embedding", True):
                available.append(info["name"])

        return available

    async def async_embed_text(self, config_subentry: dict, text: str, **kwargs) -> List[float]:
        try:
            session = async_get_clientsession(self.hass)
            url = ABaseEmbedder._format_url(
                hostname=self.client_options.get(CONF_EMBEDDING_HOST),
                port=self.client_options.get(CONF_EMBEDDING_PORT),
                ssl=self.client_options.get(CONF_EMBEDDING_SSL),
                path="/api/embed"
            )
            
            payload = {
                "model": config_subentry[CONF_EMBEDDING_MODEL],
                "input": text
            }
            
            if "keep_alive" in kwargs:
                payload["keep_alive"] = kwargs["keep_alive"]
                payload.pop("input")
            
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
                response.raise_for_status()
                data = await response.json()
                
                if "embeddings" in data and len(data["embeddings"]) > 0:
                    return data["embeddings"][0]
                else:
                    if "keep_alive" not in kwargs:
                        _logger.error(f"Unexpected response from embedding backend: {data}")
                    return []
        except Exception as e:
            _logger.error(f"Error embedding text: {e}", exc_info=True)
            return []

    async def async_embed_devices(self, config_subentry: dict, devices: List[Device]) -> List[DeviceEmbedding]:
        return [DeviceEmbedding(device=device, vector_embedding=await self.async_embed_text(config_subentry, str(device))) for device in devices]