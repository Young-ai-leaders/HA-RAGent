import asyncio
import json
import logging
from typing import Any, Dict, List, AsyncGenerator

try:
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.aiohttp_client import async_get_clientsession
except ImportError:
    from custom_components.ha_ragent.src.backends.homeassistant_mock import (
        MockHomeAssistant as HomeAssistant,
        async_get_clientsession,
    )

from custom_components.ha_ragent.src.backends.llm.base_backend import ALlmBaseBackend
from custom_components.ha_ragent.src.const import (
    CONF_CONTEXT_LENGTH,
    CONF_ENABLE_MODEL_THINKING,
    CONF_LLM_HOST,
    CONF_LLM_MODEL,
    CONF_LLM_PORT,
    CONF_LLM_SSL,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE
)
from custom_components.ha_ragent.src.models.tool import LlmTool
from custom_components.ha_ragent.src.models.model_info import ModelInfo

_logger = logging.getLogger(__name__)

class OllamaLlmBackend(ALlmBaseBackend):
    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]):
        super().__init__(hass, client_options)
        self._tags_url = ALlmBaseBackend.format_url(**self._url_base, path="/api/tags")
        self._info_url = ALlmBaseBackend.format_url(**self._url_base, path="/api/show")
        self._chat_url = ALlmBaseBackend.format_url(**self._url_base, path="/api/chat")

    @staticmethod
    def get_name() -> str:
        return f"{ALlmBaseBackend.get_name()}: Ollama"
    
    @staticmethod
    async def async_validate_connection(hass: HomeAssistant, user_input: Dict[str, Any]) -> str | None:
        try:
            session = async_get_clientsession(hass)
            
            async with session.get(
                ALlmBaseBackend.format_url(
                    hostname=user_input.get(CONF_LLM_HOST),
                    port=user_input.get(CONF_LLM_PORT),
                    ssl=user_input.get(CONF_LLM_SSL),
                    path="/api/tags"
                ),
                timeout=ALlmBaseBackend._default_timeout
            ) as response:
                return None if response.ok else f"HTTP Status {response.status}"
        except Exception as ex:
            return str(ex)
        
    async def async_get_model_info(self, model_name: str) -> ModelInfo:
        session = async_get_clientsession(self._hass)
        async with session.post(
            self._info_url,
            json={"model": model_name},
            timeout=ALlmBaseBackend._default_timeout
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
        async for _ in self.async_send_chat_request(config_subentry, [{"role": "system", "content": "Preloading model with a test embedding request."}], [], keep_alive=-1):
            pass
    
    async def async_unload_model(self, config_subentry: dict) -> None:
        async for _ in self.async_send_chat_request(config_subentry, [{"role": "system", "content": "Unloading model with a test embedding request."}], [], keep_alive=0):
            pass
    
    async def async_get_available_models(self) -> List[str]:
        session = async_get_clientsession(self._hass)
        async with session.get(
            self._tags_url,
            timeout=ALlmBaseBackend._default_timeout
        ) as response:
            response.raise_for_status()
            models_result = await response.json()

        names = [x["name"] for x in models_result.get("models", [])]
        infos = await asyncio.gather(*(self.async_get_model_info(name) for name in names), return_exceptions=True)
        available = []
        for info in infos:
            if isinstance(info, Exception):
                continue
            if info.is_tool_model:
                available.append(info.name)

        return available
    
    async def async_send_chat_request(self, config_subentry: dict, messages: List[Dict[str, str]], tools: List[LlmTool], **kwargs) -> AsyncGenerator[str, None]:
        """Send a chat request to Ollama and stream responses."""
        session = async_get_clientsession(self._hass)

        payload = {
            "model": config_subentry[CONF_LLM_MODEL],
            "stream": True,
            "think": config_subentry[CONF_ENABLE_MODEL_THINKING],
            "temperature": config_subentry[CONF_TEMPERATURE],
            "num_ctx": config_subentry[CONF_CONTEXT_LENGTH],
            "num_predict": config_subentry[CONF_MAX_TOKENS],
        }
        
        if "keep_alive" in kwargs:
            payload["keep_alive"] = kwargs["keep_alive"]
        else:
            payload["messages"] = messages

        if tools:
            payload["tools"] = [tool.to_tool_dict() for tool in tools]
            _logger.info("Added %d tools to Ollama request", len(tools))
        
        try:
            async with session.post(self._chat_url, json=payload, timeout=ALlmBaseBackend._chat_timeout) as response:
                response.raise_for_status()
                async for line in response.content:
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        
                        if "message" in data and "content" in data["message"]:
                            content = data["message"]["content"]
                            if content:
                                yield content
                        
                        if "message" in data and "tool_calls" in data["message"]:
                            tool_calls = data["message"]["tool_calls"]
                            if tool_calls:
                                _logger.debug("Received %d tool calls from Ollama", len(tool_calls))
                                for tc in tool_calls:
                                    if "function" in tc:
                                        func = tc["function"]
                                        tool_json = {
                                            "tool": func.get("name", "unknown"),
                                            "arguments": func.get("arguments", {})
                                        }
                                        yield f"\n```homeassistant\n{json.dumps(tool_json)}\n```\n"

                    except json.JSONDecodeError:
                        _logger.debug("Failed to parse Ollama response: %s", line)
                        continue
        except Exception as err:
            _logger.error("Error calling Ollama API: %s", err, exc_info=True)
            raise
