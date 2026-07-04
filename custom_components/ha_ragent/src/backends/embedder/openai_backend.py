import logging
from typing import Any, Dict, List

import aiohttp

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from ...const import (
    CONF_EMBEDDING_API_KEY,
    CONF_EMBEDDING_HOST,
    CONF_EMBEDDING_MODEL,
    CONF_EMBEDDING_PORT,
    CONF_EMBEDDING_SSL,
)
from ...models.device import Device
from ...models.device_embedding import DeviceEmbedding
from ...models.tool import LlmTool
from ...models.tool_embedding import LlmToolEmbedding

from .base_backend import ABaseEmbedder

_logger = logging.getLogger(__name__)


class OpenAICompatibleEmbedder(ABaseEmbedder):
    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]):
        super().__init__(hass, client_options)

        base = {
            "hostname": client_options.get(CONF_EMBEDDING_HOST),
            "port": client_options.get(CONF_EMBEDDING_PORT),
            "ssl": client_options.get(CONF_EMBEDDING_SSL),
        }
        self._models_url = self._format_url(**base, path="/v1/models")
        self._embeddings_url = self._format_url(**base, path="/v1/embeddings")

        self._default_timeout = aiohttp.ClientTimeout(total=5)
        self._request_timeout = aiohttp.ClientTimeout(total=30)
        self._session = async_get_clientsession(hass)

    @staticmethod
    def get_name(client_options: Dict[str, Any]):
        return "Embedder: OpenAI Compatible"

    def _headers(self, config_subentry: dict) -> Dict[str, str]:
        api_key = str(config_subentry.get(CONF_EMBEDDING_API_KEY, "")).strip()
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    @staticmethod
    async def async_validate_connection(hass: HomeAssistant, user_input: Dict[str, Any]) -> str | None:
        try:
            session = async_get_clientsession(hass)
            url = ABaseEmbedder._format_url(
                hostname=user_input.get(CONF_EMBEDDING_HOST),
                port=user_input.get(CONF_EMBEDDING_PORT),
                ssl=user_input.get(CONF_EMBEDDING_SSL),
                path="/v1/models",
            )
            headers = {"Content-Type": "application/json"}
            api_key = str(user_input.get(CONF_EMBEDDING_API_KEY, "")).strip()
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5), headers=headers) as response:
                if response.ok:
                    return None
                return f"HTTP Status {response.status}"
        except Exception as ex:
            return str(ex)

    async def async_preload_model(self, config_subentry: dict) -> None:
        _logger.debug("OpenAI-compatible embeddings do not support explicit preload")

    async def async_unload_model(self, config_subentry: dict) -> None:
        _logger.debug("OpenAI-compatible embeddings do not support explicit unload")

    async def async_get_available_models(self) -> List[str]:
        async with self._session.get(
            self._models_url,
            timeout=self._default_timeout,
            headers={"Content-Type": "application/json"},
        ) as response:
            response.raise_for_status()
            models_result = await response.json()

        model_entries = models_result.get("data", []) or models_result.get("models", [])
        available = []
        for entry in model_entries:
            model_name = entry.get("id") or entry.get("name")
            if model_name:
                available.append(model_name)

        return available

    async def _async_embed_batch(self, config_subentry: dict, inputs: list[str]) -> list[list[float]]:
        if not inputs:
            return []

        payload = {
            "model": config_subentry[CONF_EMBEDDING_MODEL],
            "input": inputs,
        }

        try:
            async with self._session.post(
                self._embeddings_url,
                json=payload,
                timeout=self._request_timeout,
                headers=self._headers(config_subentry),
            ) as response:
                response.raise_for_status()
                data = await response.json()
        except aiohttp.ClientResponseError as err:
            if err.status >= 500 and len(inputs) > 1:
                midpoint = max(1, len(inputs) // 2)
                _logger.warning(
                    "Embedding batch of %s items failed with HTTP %s; retrying in smaller chunks.",
                    len(inputs),
                    err.status,
                )
                left = await self._async_embed_batch(config_subentry, inputs[:midpoint])
                right = await self._async_embed_batch(config_subentry, inputs[midpoint:])
                return [*left, *right]
            raise

        if isinstance(data, dict) and isinstance(data.get("data"), list):
            ordered = sorted(data["data"], key=lambda item: item.get("index", 0))
            return [item.get("embedding", []) for item in ordered]

        return []

    async def async_embed_text(self, config_subentry: dict, text: str, **kwargs) -> list[float]:
        embeddings = await self._async_embed_batch(config_subentry, [text])
        return embeddings[0] if embeddings else []

    async def async_embed_object(self, object_type: type[DeviceEmbedding | LlmToolEmbedding], config_subentry: dict, objects: List[Device | LlmTool]) -> List[DeviceEmbedding | LlmToolEmbedding]:
        if not objects:
            return []

        batch_size = 32
        object_embeddings = []
        for i in range(0, len(objects), batch_size):
            chunk = objects[i:i + batch_size]
            texts = [str(obj) for obj in chunk]
            vectors = await self._async_embed_batch(config_subentry, texts)
            for obj, vec in zip(chunk, vectors):
                object_embeddings.append(object_type(obj, vector_embedding=vec))
        return object_embeddings
