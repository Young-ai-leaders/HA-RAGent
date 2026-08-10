import asyncio
from typing import Any, Dict, List
import logging

from functools import partial
from openai import AsyncOpenAI

from .base_backend import ABaseEmbedder
from ...models.device import Device
from ...models.device_embedding import DeviceEmbedding
from ...models.tool import LlmTool
from ...models.tool_embedding import LlmToolEmbedding

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from ...const import (
    CONF_EMBEDDING_API_KEY,
    CONF_EMBEDDING_MODEL,
    CONF_EMBEDDING_HOST,
    CONF_EMBEDDING_PORT,
    CONF_EMBEDDING_SSL
)

_logger = logging.getLogger(__name__)
    
class OpenAICompatibleEmbedder(ABaseEmbedder):
    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]):
        super().__init__(hass, client_options)
        self._openai_url = ABaseEmbedder.format_url(**self._url_base, path="/v1")
    
    @staticmethod
    def get_name(client_options: Dict[str, Any]):
        return "Embedder: OpenAI Compatible"

    @staticmethod
    async def async_validate_connection(hass: HomeAssistant, user_input: Dict[str, Any]) -> str | None:
        client = None
        try:
            base_url = ABaseEmbedder.format_url(
                hostname=user_input.get(CONF_EMBEDDING_HOST),
                port=user_input.get(CONF_EMBEDDING_PORT),
                ssl=user_input.get(CONF_EMBEDDING_SSL),
                path="/v1",
            )
            api_key = str(user_input.get(CONF_EMBEDDING_API_KEY, "") or "").strip()
            client = await hass.async_add_executor_job(
                partial(
                    AsyncOpenAI,
                    base_url=base_url,
                    api_key=api_key or "not-needed",
                )
            )
            await client.models.list()
            return None
        except Exception as ex:
            return str(ex)
        finally:
            if client:
                await client.close()
    
    async def _async_get_model_info(self, model_name: str) -> Dict[str, Any]:
        async with self._session.get(
            self._models_url,
            timeout=ABaseEmbedder._default_timeout,
        ) as response:
            response.raise_for_status()
            result = await response.json()

        for model in result.get("data", []):
            if model.get("id") == model_name:
                meta = model.get("meta") or {}

                return {
                    "name": model_name,
                    "supports_tools": False,
                    "is_embedding": True,
                    "embedding_size": meta.get("n_embd"),
                    "context_size": meta.get("n_ctx_train"),
                    "parameters": meta.get("n_params"),
                    "size": meta.get("size"),
                }

        raise ValueError(f"Model not found: {model_name}")

    async def async_preload_model(self, config_subentry: dict) -> None:
        _logger.info("Preloading not supported for OpenAI Compatible Embedder backend.")

    async def async_unload_model(self, config_subentry: dict) -> None:
        _logger.info("Unloading not supported for OpenAI Compatible Embedder backend.")

    async def async_get_available_models(self) -> List[str]:
        async with self._session.get(
            self._models_url,
            timeout=ABaseEmbedder._default_timeout,
        ) as response:
            response.raise_for_status()
            result = await response.json()

        return [
            model["id"]
            for model in result.get("data", [])
            if model.get("id")
        ]

    async def _async_embed_batch(
        self,
        config_subentry: dict,
        inputs: List[str],
    ) -> List[List[float]]:
        if not inputs:
            return []

        payload = {
            "model": config_subentry[CONF_EMBEDDING_MODEL],
            "input": inputs,
            "encoding_format": "float",
        }

        async with self._session.post(
            self._embed_url,
            json=payload,
            timeout=ABaseEmbedder._default_timeout,
        ) as response:
            response.raise_for_status()
            result = await response.json()

        # OpenAI-compatible response:
        #
        # {
        #     "data": [
        #         {
        #             "embedding": [...],
        #             "index": 0,
        #             "object": "embedding"
        #         }
        #     ],
        #     ...
        # }

        data = result.get("data", [])

        # Do not blindly trust response ordering.
        data.sort(key=lambda item: item.get("index", 0))

        return [
            item["embedding"]
            for item in data
            if item.get("embedding") is not None
        ]

    async def async_embed_text(self, config_subentry: dict, text: str, **kwargs) -> List[float]:
        embeddings = await self._async_embed_batch(config_subentry, [text])
        return embeddings[0] if embeddings else []

    async def async_embed_object(self, object_type: type[DeviceEmbedding | LlmToolEmbedding], config_subentry: dict, objects: List[Device | LlmTool]) -> List[DeviceEmbedding | LlmToolEmbedding]:
        if not objects:
            return []

        batch_size = 32
        object_embeddings = []

        for i in range(0, len(objects), batch_size):
            chunk = objects[i : i + batch_size]

            texts = [
                str(obj)
                for obj in chunk
            ]

            vectors = await self._async_embed_batch(
                config_subentry,
                texts,
            )

            if len(vectors) != len(chunk):
                raise RuntimeError(
                    "llama.cpp returned an unexpected number "
                    f"of embeddings: expected {len(chunk)}, "
                    f"received {len(vectors)}"
                )

            for obj, vector in zip(chunk, vectors):
                object_embeddings.append(
                    object_type(
                        obj,
                        vector_embedding=vector,
                    )
                )

        return object_embeddings