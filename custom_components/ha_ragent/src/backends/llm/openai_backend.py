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
        self._models_url = self._format_url(**base, path="/v1/models")
        self._chat_url = self._format_url(**base, path="/v1/chat/completions")

        self._default_timeout = aiohttp.ClientTimeout(total=5)
        self._chat_timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)

        self._session = async_get_clientsession(hass)

    @staticmethod
    def get_name(client_options: Dict[str, Any]):
        return "LLM: OpenAI Compatible"

    def _headers(self, config_subentry: dict) -> Dict[str, str]:
        api_key = str(config_subentry.get(CONF_LLM_API_KEY, "")).strip()
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

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
                if response.ok:
                    return None
                return f"HTTP Status {response.status}"
        except Exception as ex:
            return str(ex)

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
                response.raise_for_status()

                async for raw_line in response.content:
                    if not raw_line:
                        continue

                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line or not line.startswith("data:"):
                        continue

                    data_text = line.removeprefix("data:").strip()
                    if data_text == "[DONE]":
                        break

                    try:
                        data = json.loads(data_text)
                    except json.JSONDecodeError:
                        _logger.debug("Failed to parse OpenAI-compatible stream line: %s", line)
                        continue

                    choices = data.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {}) or {}
                    content = delta.get("content")
                    if content:
                        yield content

                    for tool_call in delta.get("tool_calls", []) or []:
                        index = tool_call.get("index", 0)
                        current = pending_tool_calls.setdefault(index, {"function": {"name": "", "arguments": ""}})
                        function_data = tool_call.get("function", {}) or {}
                        if function_data.get("name"):
                            current["function"]["name"] = function_data["name"]
                        if function_data.get("arguments"):
                            current["function"]["arguments"] += function_data["arguments"]

            for index in sorted(pending_tool_calls):
                yield self._make_tool_result_block(pending_tool_calls[index])

        except Exception as err:
            _logger.error("Error calling OpenAI-compatible API: %s", err, exc_info=True)
            raise
