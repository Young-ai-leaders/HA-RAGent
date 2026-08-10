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
    def get_name() -> str:
        return f"{ABaseEmbedder.get_name()}: OpenAI Compatible"

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

    async def _async_create_client(self) -> AsyncOpenAI:
        return await self.hass.async_add_executor_job(
            partial(
                AsyncOpenAI,
                base_url=self._openai_url,
                api_key=self._api_key or "not-needed",
            )
        )

    async def _async_get_model_info(
        self,
        model_name: str,
    ) -> Dict[str, Any]:
        client = await self._async_create_client()

        try:
            model = await client.models.retrieve(model_name)

            model_data = model.model_dump()

            meta = model_data.get("meta") or {}

            return {
                "name": model.id,
                "supports_tools": False,
                "is_embedding": True,
                "embedding_size": meta.get("n_embd"),
                "context_size": meta.get("n_ctx_train"),
                "parameters": meta.get("n_params"),
                "size": meta.get("size"),
            }

        finally:
            await client.close()

    async def async_preload_model(self, config_subentry: dict) -> None:
        _logger.info("Preloading not supported for OpenAI Compatible Embedder backend.")

    async def async_unload_model(self, config_subentry: dict) -> None:
        _logger.info("Unloading not supported for OpenAI Compatible Embedder backend.")

    async def async_get_available_models(self) -> List[str]:
        client = await self._async_create_client()

        try:
            result = await client.models.list()

            return [
                model.id
                for model in result.data
                if model.id
            ]

        finally:
            await client.close()

    async def _async_embed_batch(
        self,
        config_subentry: dict,
        inputs: List[str],
    ) -> List[List[float]]:
        if not inputs:
            return []

        client = await self._async_create_client()

        try:
            response = await client.embeddings.create(
                model=config_subentry[CONF_EMBEDDING_MODEL],
                input=inputs,
                encoding_format="float",
            )

            # OpenAI-compatible embedding responses contain
            # an index for every input. Sort explicitly instead
            # of relying on response ordering.
            data = sorted(
                response.data,
                key=lambda item: item.index,
            )

            return [
                list(item.embedding)
                for item in data
            ]

        finally:
            await client.close()

    async def async_embed_text(
        self,
        config_subentry: dict,
        text: str,
        **kwargs,
    ) -> List[float]:
        embeddings = await self._async_embed_batch(
            config_subentry,
            [text],
        )

        return embeddings[0] if embeddings else []

    async def async_embed_object(
        self,
        object_type: type[
            DeviceEmbedding | LlmToolEmbedding
        ],
        config_subentry: dict,
        objects: List[Device | LlmTool],
    ) -> List[
        DeviceEmbedding | LlmToolEmbedding
    ]:
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
                    "OpenAI-compatible embedding server "
                    "returned an unexpected number of embeddings: "
                    f"expected {len(chunk)}, "
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
