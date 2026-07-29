import json
import logging
from typing import Any, Dict, List, AsyncGenerator

import aiohttp

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from custom_components.ha_ragent.src.models.tool import LlmTool

from ...const import (
    CONF_LLM_API_KEY,
    CONF_LLM_HOST,
    CONF_LLM_MODEL,
    CONF_LLM_PORT,
    CONF_LLM_SSL,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
)
from .base_backend import ALlmBaseBackend

_logger = logging.getLogger(__name__)


class OpenAICompatibleBackend(ALlmBaseBackend):
    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]):
            super().__init__(hass, client_options)
    
            base = {
                "hostname": client_options.get(CONF_LLM_HOST),
                "port": client_options.get(CONF_LLM_PORT),
                "ssl": client_options.get(CONF_LLM_SSL),
            }
            self._tags_url = self._format_url(**base, path="/api/tags")
            self._info_url = self._format_url(**base, path="/api/show")
            self._chat_url = self._format_url(**base, path="/api/chat")
    
            self._default_timeout = aiohttp.ClientTimeout(total=5)
            self._chat_timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)
    
            self._session = async_get_clientsession(hass)
    
    @staticmethod
    def get_name(client_options: Dict[str, Any]):
        return "LLM: OpenAI Compatible"
        
    @staticmethod
    async def async_validate_connection(hass: HomeAssistant, user_input: Dict[str, Any]) -> str | None:
        try:
            session = async_get_clientsession(hass)
            url = ALlmBaseBackend._format_url(
                hostname=user_input.get(CONF_LLM_HOST),
                port=user_input.get(CONF_LLM_PORT),
                ssl=user_input.get(CONF_LLM_SSL),
                path="/v1/models",
            )
            headers = {"Content-Type": "application/json"}
            api_key = str(user_input.get(CONF_LLM_API_KEY, "")).strip()
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5), headers=headers) as response:
                return None if response.ok else f"HTTP Status {response.status}"
        except Exception as ex:
            return str(ex)
<<<<<<< HEAD
            
        async def _async_get_model_info(self, model_name: str) -> Dict[str, Any]:
            session = async_get_clientsession(self.hass)
            async with session.post(
                self._info_url,
                json={"model": model_name},
                timeout=self._default_timeout,
                headers={},
            ) as response:
=======

    async def async_preload_model(self, config_subentry: dict) -> None:
        _logger.debug("OpenAI-compatible chat models do not support explicit preload")

    async def async_unload_model(self, config_subentry: dict) -> None:
        _logger.debug("OpenAI-compatible chat models do not support explicit unload")

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

    def _normalize_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        normalized = []
        for message in messages:
            role = str(message.get("role", "user")).lower()
            content = message.get("content", "")
            normalized_message: Dict[str, Any] = {"role": role, "content": content}

            if role == "tool" and message.get("tool_call_id"):
                normalized_message["tool_call_id"] = message["tool_call_id"]

            normalized.append(normalized_message)
        return normalized

    def _make_tool_result_block(self, tool_call: Dict[str, Any]) -> str:
        function_data = tool_call.get("function", {})
        arguments = function_data.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass

        tool_json = {
            "tool": function_data.get("name", "unknown"),
            "arguments": arguments,
        }
        return f"\n```homeassistant\n{json.dumps(tool_json)}\n```\n"

    def _collect_tool_calls(
        self,
        pending_tool_calls: dict[int, Dict[str, Any]],
        tool_calls: list[dict[str, Any]] | None,
    ) -> None:
        for tool_call in tool_calls or []:
            index = tool_call.get("index", 0)
            current = pending_tool_calls.setdefault(index, {"function": {"name": "", "arguments": ""}})
            function_data = tool_call.get("function", {}) or {}

            if function_data.get("name"):
                current["function"]["name"] = function_data["name"]

            arguments = function_data.get("arguments")
            if isinstance(arguments, str):
                current["function"]["arguments"] += arguments
            elif arguments:
                current["function"]["arguments"] = json.dumps(arguments)

    async def async_send_chat_request(self, config_subentry: dict, messages: List[Dict[str, str]], tools: List[LlmTool], **kwargs) -> AsyncGenerator[str, None]:
        payload: Dict[str, Any] = {
            "model": config_subentry[CONF_LLM_MODEL],
            "stream": True,
            "messages": self._normalize_messages(messages),
        }

        temperature = config_subentry.get(CONF_TEMPERATURE)
        if temperature is not None:
            payload["temperature"] = temperature

        max_tokens = config_subentry.get(CONF_MAX_TOKENS)
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if tools:
            payload["tools"] = [tool.to_tool_dict() for tool in tools]
            _logger.info("Added %d tools to OpenAI-compatible request", len(tools))

        if "keep_alive" in kwargs:
            _logger.debug("Ignoring keep_alive for OpenAI-compatible backend")

        pending_tool_calls: dict[int, Dict[str, Any]] = {}

        try:
            async with self._session.post(self._chat_url, json=payload, timeout=self._chat_timeout, headers=self._headers(config_subentry)) as response:
>>>>>>> origin/add_more_assistant_features
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
            async for _ in self.async_send_chat_request(config_subentry, [{"role": "system", "content": "Preloading model with a test embedding request."}], [], keep_alive=-1):
                pass
        
        async def async_unload_model(self, config_subentry: dict) -> None:
            async for _ in self.async_send_chat_request(config_subentry, [{"role": "system", "content": "Unloading model with a test embedding request."}], [], keep_alive=0):
                pass
        
        async def async_get_available_models(self) -> List[str]:
            session = async_get_clientsession(self.hass)
            async with session.get(
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
                if info.get("supports_tools", True):
                    available.append(info["name"])
    
            return available
        
    
        async def async_send_chat_request(self, config_subentry: dict, messages: List[Dict[str, str]], tools: List[LlmTool], **kwargs) -> AsyncGenerator[str, None]:
            """Send a chat request to Ollama and stream responses."""
            session = async_get_clientsession(self.hass)
    
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
                async with session.post(self._chat_url, json=payload, timeout=self._chat_timeout) as response:
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
    
    # def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]):
    #     super().__init__(hass, client_options)

    #     base = {
    #         "hostname": client_options.get(CONF_LLM_HOST),
    #         "port": client_options.get(CONF_LLM_PORT),
    #         "ssl": client_options.get(CONF_LLM_SSL),
    #     }
    #     self._models_url = self._format_url(**base, path="/v1/models")
    #     self._chat_url = self._format_url(**base, path="/v1/chat/completions")

    #     self._default_timeout = aiohttp.ClientTimeout(total=5)
    #     self._chat_timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)

    #     self._session = async_get_clientsession(hass)

    # def _headers(self, config_subentry: dict) -> Dict[str, str]:
    #     api_key = str(config_subentry.get(CONF_LLM_API_KEY, "")).strip()
    #     headers = {"Content-Type": "application/json"}
    #     if api_key:
    #         headers["Authorization"] = f"Bearer {api_key}"
    #     return headers


<<<<<<< HEAD

    # async def async_preload_model(self, config_subentry: dict) -> None:
    #     _logger.debug("OpenAI-compatible chat models do not support explicit preload")
=======
                    choice = choices[0]
                    delta = choice.get("delta", {}) or {}
                    content = delta.get("content")
                    if content:
                        yield content

                    self._collect_tool_calls(pending_tool_calls, delta.get("tool_calls"))

                    message = choice.get("message", {}) or {}
                    message_content = message.get("content")
                    if message_content and not content:
                        yield message_content

                    self._collect_tool_calls(pending_tool_calls, message.get("tool_calls"))
>>>>>>> origin/add_more_assistant_features

    # async def async_unload_model(self, config_subentry: dict) -> None:
    #     _logger.debug("OpenAI-compatible chat models do not support explicit unload")

<<<<<<< HEAD
    # async def async_get_available_models(self) -> List[str]:
    #     async with self._session.get(
    #         self._models_url,
    #         timeout=self._default_timeout,
    #         headers={"Content-Type": "application/json"},
    #     ) as response:
    #         response.raise_for_status()
    #         models_result = await response.json()

    #     model_entries = models_result.get("data", []) or models_result.get("models", [])
    #     available = []
    #     for entry in model_entries:
    #         model_name = entry.get("id") or entry.get("name")
    #         if model_name:
    #             available.append(model_name)

    #     return available

    # def _normalize_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    #     normalized = []
    #     for message in messages:
    #         role = str(message.get("role", "user")).lower()
    #         content = message.get("content", "")
    #         normalized_message: Dict[str, Any] = {"role": role, "content": content}

    #         if role == "tool" and message.get("tool_call_id"):
    #             normalized_message["tool_call_id"] = message["tool_call_id"]

    #         normalized.append(normalized_message)
    #     return normalized

    # def _make_tool_result_block(self, tool_call: Dict[str, Any]) -> str:
    #     function_data = tool_call.get("function", {})
    #     arguments = function_data.get("arguments", {})
    #     if isinstance(arguments, str):
    #         try:
    #             arguments = json.loads(arguments)
    #         except json.JSONDecodeError:
    #             pass

    #     tool_json = {
    #         "tool": function_data.get("name", "unknown"),
    #         "arguments": arguments,
    #     }
    #     return f"\n```homeassistant\n{json.dumps(tool_json)}\n```\n"

    # async def async_send_chat_request(self, config_subentry: dict, messages: List[Dict[str, str]], tools: List[LlmTool], **kwargs) -> AsyncGenerator[str, None]:
    #     payload: Dict[str, Any] = {
    #         "model": config_subentry[CONF_LLM_MODEL],
    #         "stream": True,
    #         "messages": self._normalize_messages(messages),
    #     }

    #     temperature = config_subentry.get(CONF_TEMPERATURE)
    #     if temperature is not None:
    #         payload["temperature"] = temperature

    #     max_tokens = config_subentry.get(CONF_MAX_TOKENS)
    #     if max_tokens is not None:
    #         payload["max_tokens"] = max_tokens

    #     if tools:
    #         payload["tools"] = [tool.to_tool_dict() for tool in tools]
    #         _logger.info("Added %d tools to OpenAI-compatible request", len(tools))

    #     if "keep_alive" in kwargs:
    #         _logger.debug("Ignoring keep_alive for OpenAI-compatible backend")

    #     pending_tool_calls: dict[int, Dict[str, Any]] = {}

    #     try:
    #         async with self._session.post(self._chat_url, json=payload, timeout=self._chat_timeout, headers=self._headers(config_subentry)) as response:
    #             response.raise_for_status()

    #             async for raw_line in response.content:
    #                 if not raw_line:
    #                     continue

    #                 line = raw_line.decode("utf-8", errors="ignore").strip()
    #                 if not line or not line.startswith("data:"):
    #                     continue

    #                 data_text = line.removeprefix("data:").strip()
    #                 if data_text == "[DONE]":
    #                     break

    #                 try:
    #                     data = json.loads(data_text)
    #                 except json.JSONDecodeError:
    #                     _logger.debug("Failed to parse OpenAI-compatible stream line: %s", line)
    #                     continue

    #                 choices = data.get("choices", [])
    #                 if not choices:
    #                     continue

    #                 delta = choices[0].get("delta", {}) or {}
    #                 content = delta.get("content")
    #                 if content:
    #                     yield content

    #                 for tool_call in delta.get("tool_calls", []) or []:
    #                     index = tool_call.get("index", 0)
    #                     current = pending_tool_calls.setdefault(index, {"function": {"name": "", "arguments": ""}})
    #                     function_data = tool_call.get("function", {}) or {}
    #                     if function_data.get("name"):
    #                         current["function"]["name"] = function_data["name"]
    #                     if function_data.get("arguments"):
    #                         current["function"]["arguments"] += function_data["arguments"]

    #         for index in sorted(pending_tool_calls):
    #             yield self._make_tool_result_block(pending_tool_calls[index])

    #     except Exception as err:
    #         _logger.error("Error calling OpenAI-compatible API: %s", err, exc_info=True)
    #         raise
=======
        except Exception as err:
            _logger.error("Error calling OpenAI-compatible API: %s", err, exc_info=True)
            raise
>>>>>>> origin/add_more_assistant_features
