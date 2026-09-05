import asyncio
import logging
from typing import Any, Dict, List

from custom_components.ha_ragent.src.models.model_info import ModelInfo
from custom_components.ha_ragent.src.models.base.embeddable_model import EmbeddableModel
from custom_components.ha_ragent.src.models.base.embedding_record import EmbeddingRecord

try:
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.aiohttp_client import async_get_clientsession
except ImportError:
    from custom_components.ha_ragent.src.mock import (
        MockHomeAssistant as HomeAssistant,
        async_get_clientsession,
    )

from custom_components.ha_ragent.src.const import (
    CONF_EMBEDDING_HOST,
    CONF_EMBEDDING_PORT,
    CONF_EMBEDDING_SSL,
    CONF_EMBEDDING_MODEL,
    RAGENT_EMBEDDING_BATCH_SIZE,
    RAGENT_EMBEDDING_TRUNCATE_MAX_CHARS,
)
from custom_components.ha_ragent.src.backends.embedder.base_backend import ABaseEmbedder

_logger = logging.getLogger(__name__)
    
class OllamaEmbedder(ABaseEmbedder):
    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]):
        super().__init__(hass, client_options)
        self._tags_url = ABaseEmbedder.format_url(**self._url_base, path="/api/tags")
        self._info_url = ABaseEmbedder.format_url(**self._url_base, path="/api/show")
        self._embed_url = ABaseEmbedder.format_url(**self._url_base, path="/api/embed")

    @staticmethod
    def get_name() -> str:
        return f"{ABaseEmbedder.get_name()}: Ollama"

    @staticmethod
    async def async_validate_connection(hass: HomeAssistant, user_input: Dict[str, Any]) -> str | None:
        try:
            session = async_get_clientsession(hass)
            
            async with session.get(
                ABaseEmbedder.format_url(
                    hostname=user_input.get(CONF_EMBEDDING_HOST),
                    port=user_input.get(CONF_EMBEDDING_PORT),
                    ssl=user_input.get(CONF_EMBEDDING_SSL),
                    path="/api/tags"
                ),
                timeout=ABaseEmbedder._default_timeout
            ) as response:
                return None if response.ok else f"HTTP Status {response.status}"
        except Exception as ex:
            return str(ex)
    
    async def async_get_model_info(self, model_name: str) -> ModelInfo:
        session = async_get_clientsession(self._hass)
        async with session.post(
            self._info_url,
            json={"model": model_name},
            timeout=ABaseEmbedder._default_timeout
        ) as response:
            response.raise_for_status()
            model_result = await response.json()

        capabilities = model_result.get("capabilities", [])
        is_tool_model = "tools" in capabilities
        is_embedding_model = "embedding" in capabilities

        return ModelInfo(
            name=model_name,
            context_size=None,
            is_tool_model=is_tool_model,
            is_embedding_model=is_embedding_model
        )

    async def async_preload_model(self, config_subentry: dict) -> None:
        await self.async_embed_text(config_subentry, "Preloading model with a test embedding request.", keep_alive=-1)  
    
    async def async_unload_model(self, config_subentry: dict) -> None:
        await self.async_embed_text(config_subentry, "Unloading model with a test embedding request.", keep_alive=0)
    
    async def async_get_available_models(self) -> List[str]:
        session = async_get_clientsession(self._hass)
        async with session.get(
            self._tags_url,
            timeout=ABaseEmbedder._default_timeout,
        ) as response:
            response.raise_for_status()
            models_result = await response.json()

        names = [x["name"] for x in models_result.get("models", [])]
        infos = await asyncio.gather(*(self.async_get_model_info(name) for name in names), return_exceptions=True)
        available = []
        for info in infos:
            if isinstance(info, Exception):
                continue
            if info.is_embedding_model:
                available.append(info.name)

        return available

    async def _async_embed_batch(self, config_subentry: dict, inputs: list[str], keep_alive: int | None = None) -> list[list[float]]:
        payload = {"model": config_subentry[CONF_EMBEDDING_MODEL]}
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        else:
            payload["input"] = [text[:RAGENT_EMBEDDING_TRUNCATE_MAX_CHARS] for text in inputs]

        session = async_get_clientsession(self._hass)
        async with session.post(self._embed_url, json=payload, timeout=ABaseEmbedder._default_timeout) as response:
            response.raise_for_status()
            data = await response.json()
            return data.get("embeddings", [])

    async def async_embed_text(self, config_subentry: dict, text: str, **kwargs) -> list[float]:
        keep_alive = kwargs.get("keep_alive")
        embeddings = await self._async_embed_batch(config_subentry, [text], keep_alive=keep_alive)
        return embeddings[0] if embeddings else []

    async def async_embed_object(self, config_subentry: dict, objects: List[EmbeddableModel]) -> List[EmbeddingRecord]:
        if not objects:
            return []

        object_embeddings: List[EmbeddingRecord] = []
        for i in range(0, len(objects), RAGENT_EMBEDDING_BATCH_SIZE):
            chunk = objects[i:i + RAGENT_EMBEDDING_BATCH_SIZE]
            texts = [obj.to_embedding_text() for obj in chunk]
            vectors = await self._async_embed_batch(config_subentry, texts)
            object_embeddings.extend(self.build_embedding_records(chunk, vectors))
        return object_embeddings
