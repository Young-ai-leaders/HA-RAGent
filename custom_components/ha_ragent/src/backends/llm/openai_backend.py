import json
import logging
from functools import partial
from typing import Any, AsyncGenerator, Dict, List

from openai import AsyncOpenAI

from homeassistant.core import HomeAssistant

from custom_components.ha_ragent.src.models.tool import LlmTool

from .base_backend import ALlmBaseBackend
from ...const import (
    CONF_ENABLE_MODEL_THINKING,
    CONF_LLM_API_KEY,
    CONF_LLM_HOST,
    CONF_LLM_MODEL,
    CONF_LLM_PORT,
    CONF_LLM_SSL,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
)

_logger = logging.getLogger(__name__)

class OpenAICompatibleBackend(ALlmBaseBackend):
    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]):
        super().__init__(hass, client_options)
        self._openai_url = ALlmBaseBackend.format_url(**self._url_base, path="/v1")

    @staticmethod
    def get_name(client_options: Dict[str, Any]):
        return "LLM: OpenAI Compatible"

    @staticmethod
    async def async_validate_connection(
        hass: HomeAssistant, user_input: Dict[str, Any]
    ) -> str | None:
        client = None
        try:
            base_url = ALlmBaseBackend.format_url(
                hostname=user_input.get(CONF_LLM_HOST),
                port=user_input.get(CONF_LLM_PORT),
                ssl=user_input.get(CONF_LLM_SSL),
                path="/v1",
            )
            api_key = str(user_input.get(CONF_LLM_API_KEY, "") or "").strip()
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

    async def async_get_model_info(self, model_name: str) -> Dict[str, Any]:
        client = await self._async_create_client()

        try:
            model = await client.models.retrieve(model_name)
            return model.model_dump()
        finally:
            await client.close()

    async def async_preload_model(self, config_subentry: dict) -> None:
        _logger.info("Preloading not supported for OpenAI Compatible LLM backend.")

    async def async_unload_model(self, config_subentry: dict) -> None:
        _logger.info("Unloading not supported for OpenAI Compatible LLM backend.")

    async def async_get_available_models(self) -> List[str]:
        client = await self._async_create_client()

        try:
            result = await client.models.list()
            return [model.id for model in result.data if model.id]
        finally:
            await client.close()

    async def async_send_chat_request(
        self,
        config_subentry: dict,
        messages: List[Dict[str, str]],
        tools: List[LlmTool],
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        client = await self._async_create_client()

        request: Dict[str, Any] = {
            "model": config_subentry[CONF_LLM_MODEL],
            "messages": messages,
            "stream": True,
            "temperature": config_subentry[CONF_TEMPERATURE],
            "max_tokens": config_subentry[CONF_MAX_TOKENS],
        }

        if tools:
            request["tools"] = self.convert_tools_to_model_format(tools)
            request["tool_choice"] = "auto"

            # llama.cpp-specific request fields go through the SDK's
            # extension body so the OpenAI client still validates standard
            # Chat Completions fields.
            request["extra_body"] = {
                "parse_tool_calls": True,
                "chat_template_kwargs": {
                    "enable_thinking": bool(
                        config_subentry[CONF_ENABLE_MODEL_THINKING]
                    ),
                },
            }
            _logger.debug(f"Added {len(tools)} tools to OpenAI-compatible request")

        try:
            pending_tool_calls: Dict[int, Dict[str, str]] = {}
            stream = await client.chat.completions.create(**request)

            async for chunk in stream:
                for choice in chunk.choices:
                    delta = choice.delta

                    if delta.content:
                        yield delta.content

                    reasoning_content = getattr(delta, "reasoning_content", None)
                    if reasoning_content:
                        _logger.debug(f"llama.cpp reasoning: {reasoning_content}")

                    for tool_call in delta.tool_calls or []:
                        index = tool_call.index or 0
                        function = tool_call.function
                        pending = pending_tool_calls.setdefault(
                            index,
                            {"name": "", "arguments": ""},
                        )

                        if function and function.name:
                            pending["name"] = function.name
                        if function and function.arguments:
                            pending["arguments"] += function.arguments

            # Tool-call arguments are often split across many SSE chunks.
            # Emit only after the complete JSON document has been assembled.
            for pending in pending_tool_calls.values():
                if not pending["name"]:
                    continue

                try:
                    arguments = json.loads(pending["arguments"] or "{}")
                except json.JSONDecodeError:
                    _logger.warning(f"Could not decode complete tool-call arguments for {pending['name']}: {pending['arguments']}")
                    continue

                tool_json = {
                    "tool": pending["name"],
                    "arguments": arguments,
                }
                yield (
                    "\n```homeassistant\n"
                    f"{json.dumps(tool_json)}"
                    "\n```\n"
                )

        except Exception as err:
            _logger.error(f"Error calling llama.cpp API through OpenAI client: {err}", exc_info=True)
            raise
        finally:
            await client.close()
