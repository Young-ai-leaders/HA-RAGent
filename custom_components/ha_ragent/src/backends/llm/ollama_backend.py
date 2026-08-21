import asyncio
import json
import logging
from typing import Any, Dict, List, AsyncGenerator

try:
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.aiohttp_client import async_get_clientsession
except ImportError:
    from custom_components.ha_ragent.src.mock import (
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
from custom_components.ha_ragent.src.models.chat_message import ChatMessage
from custom_components.ha_ragent.src.const import RAGENT_CHAT_TRUNCATE_MAX_CHARS, RAGENT_CHAT_TRUNCATE_RETRIES

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

    def format_messages_for_backend(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """Convert canonical history messages to Ollama chat format."""
        prepared: List[ChatMessage] = []
        for message in messages:
            item = dict(message)
            if item.get("role") == "assistant" and item.get("tool_calls"):
                item["tool_calls"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool_call["function"]["name"],
                            "arguments": tool_call["function"]["arguments"],
                        },
                    }
                    for tool_call in item["tool_calls"]
                ]
            if item.get("role") == "tool":
                item.pop("tool_call_id", None)
                if not isinstance(item.get("content"), str):
                    item["content"] = json.dumps(item.get("content"), ensure_ascii=False, default=str)
            prepared.append(item)
        return prepared
    
    async def _async_send_chat_request_once(self, config_subentry: dict, messages: List[ChatMessage], tools: List[LlmTool], **kwargs) -> AsyncGenerator[str, None]:
        """Send one Ollama request while preserving streaming output."""
        session = async_get_clientsession(self._hass)
        emitted = kwargs.pop("_emitted", None)
        if emitted is None:
            emitted = {"value": False}

        payload = {
            "model": config_subentry[CONF_LLM_MODEL],
            "stream": True,
            "think": config_subentry[CONF_ENABLE_MODEL_THINKING],
            "options": {
                "temperature": config_subentry[CONF_TEMPERATURE],
                "num_ctx": config_subentry[CONF_CONTEXT_LENGTH],
                "num_predict": config_subentry[CONF_MAX_TOKENS],
            },
        }
        
        if "keep_alive" in kwargs:
            payload["keep_alive"] = kwargs["keep_alive"]
        else:
            payload["messages"] = self.format_messages_for_backend(messages)

        if tools:
            payload["tools"] = [tool.to_tool_dict() for tool in tools]
            tool_names, required_tool_names = self.split_tool_names(tools)
            _logger.debug(f"Added {len(tools)} tools to Ollama request: tools={tool_names}, required_tools={required_tool_names}")
        
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
                                emitted["value"] = True
                                yield content
                        
                        if "message" in data and "tool_calls" in data["message"]:
                            tool_calls = data["message"]["tool_calls"]
                            if tool_calls:
                                emitted["value"] = True
                                _logger.debug(f"LLM tool calls received from Ollama: {tool_calls}")
                                for tc in tool_calls:
                                    if "function" in tc:
                                        func = tc["function"]
                                        tool_json = {
                                            "tool": func.get("name", "unknown"),
                                            "arguments": func.get("arguments", {})
                                        }
                                        yield f"\n```homeassistant\n{json.dumps(tool_json)}\n```\n"

                    except json.JSONDecodeError:
                        _logger.debug(f"Failed to parse Ollama response: {line}")
                        continue
        except Exception as err:
            _logger.error("Error calling Ollama API: %s", err, exc_info=True)
            raise
        return

    async def async_send_chat_request(self, config_subentry: dict, messages: List[ChatMessage], tools: List[LlmTool], **kwargs) -> AsyncGenerator[str, None]:
        """Send a chat request to Ollama and retry with truncation when empty."""
        current_messages = messages
        max_chars = RAGENT_CHAT_TRUNCATE_MAX_CHARS

        for attempt in range(RAGENT_CHAT_TRUNCATE_RETRIES + 1):
            emitted = {"value": False}
            async for chunk in self._async_send_chat_request_once(config_subentry, current_messages, tools, _emitted=emitted, **kwargs):
                yield chunk

            if emitted["value"] or not messages:
                return

            if attempt == RAGENT_CHAT_TRUNCATE_RETRIES:
                return

            max_chars //= 2
            current_messages = self.truncate_messages(messages, max_chars)
            _logger.warning(f"Ollama returned an empty response. Retrying with messages limited to {max_chars} characters.")