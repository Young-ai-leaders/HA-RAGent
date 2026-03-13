import asyncio
from typing import Any, Dict, List
import logging
import aiohttp

from .base_backend import ABaseEmbedder
from ...models.device import Device
from ...models.device_embedding import DeviceEmbedding
from ...models.tool import LlmTool
from ...models.tool_embedding import LlmToolEmbedding

from homeassistant.core import HomeAssistant
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

        base = {
            "hostname": client_options.get(CONF_EMBEDDING_HOST),
            "port": client_options.get(CONF_EMBEDDING_PORT),
            "ssl": client_options.get(CONF_EMBEDDING_SSL),
        }
        self._tags_url = self._format_url(**base, path="/api/tags")
        self._info_url = self._format_url(**base, path="/api/show")
        self._embed_url = self._format_url(**base, path="/api/embed")

        self._default_timeout = aiohttp.ClientTimeout(total=5)
        self._embed_timeout = aiohttp.ClientTimeout(total=30)

        self._session = async_get_clientsession(hass)
    
    @staticmethod
    def get_name(client_options: Dict[str, Any]):
        return "Embedder: Ollama"

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
                    path="/api/tags"
                ),
                timeout=aiohttp.ClientTimeout(total=5),
                headers=headers
            )
            return None if response.ok else f"HTTP Status {response.status}"
        except Exception as ex:
            return str(ex)
    
    async def _async_get_model_info(self, model_name: str) -> Dict[str, Any]:
        async with self._session.post(
            self._info_url,
            json={"model": model_name},
            timeout=self._default_timeout,
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
        async with self._session.get(
            self._tags_url,
            timeout=self._default_timeout,
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

    async def _async_embed_batch(self, config_subentry: dict, inputs: list[str], keep_alive: int | None = None) -> list[list[float]]:
        payload = {"model": config_subentry[CONF_EMBEDDING_MODEL]}
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        else:
            payload["input"] = inputs

        async with self._session.post(self._embed_url, json=payload, timeout=self._embed_timeout) as response:
            response.raise_for_status()
            data = await response.json()
            return data.get("embeddings", [])

    async def async_embed_text(self, config_subentry: dict, text: str, **kwargs) -> list[float]:
        keep_alive = kwargs.get("keep_alive")
        embeddings = await self._async_embed_batch(config_subentry, [text], keep_alive=keep_alive)
        return embeddings[0] if embeddings else []

    async def async_embed_object(self, config_subentry: dict, devices: List[Device | LlmTool]) -> List[DeviceEmbedding | LlmToolEmbedding]:
        if not devices:
            return []

        batch_size = 32
        device_embeddings = []
        for i in range(0, len(devices), batch_size):
            chunk = devices[i:i + batch_size]
            texts = [str(d) for d in chunk]
            vectors = await self._async_embed_batch(config_subentry, texts)
            for device, vec in zip(chunk, vectors):
                device_embeddings.append(DeviceEmbedding(device=device, vector_embedding=vec))
        return device_embeddings