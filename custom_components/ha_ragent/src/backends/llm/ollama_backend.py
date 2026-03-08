import aiohttp
import json
import logging
from typing import Any, Dict, List, AsyncGenerator

from homeassistant.core import HomeAssistant
from homeassistant.helpers.llm import APIInstance
from homeassistant.helpers.aiohttp_client import async_get_clientsession

try:
    from voluptuous_openapi import convert
    HAS_VOLUPTUOUS_OPENAPI = True
except ImportError:
    HAS_VOLUPTUOUS_OPENAPI = False
    convert = None

from .base_backend import ALlmBaseBackend
from ...const import (
    CONF_LLM_MODEL,
    CONF_LLM_HOST,
    CONF_LLM_PORT,
    CONF_LLM_SSL,
    CONF_TEMPERATURE,
    CONF_MAX_TOKENS,
)

_logger = logging.getLogger(__name__)

class OllamaBackend(ALlmBaseBackend):
    def __init__(self, hass: HomeAssistant, client_options: dict[str, Any]):
        super().__init__(hass, client_options)

    @staticmethod
    def get_name(client_options: Dict[str, Any]):
        return f"LLM Backend: Ollama"
    
    @staticmethod
    async def async_validate_connection(hass: HomeAssistant, user_input: Dict[str, Any]) -> str | None:
        headers = {}
        try:
            session = async_get_clientsession(hass)
            response = await session.get(
                ALlmBaseBackend._format_url(
                    hostname=user_input.get(CONF_LLM_HOST),
                    port=user_input.get(CONF_LLM_PORT),
                    ssl=user_input.get(CONF_LLM_SSL),
                    path=f"/api/tags"
                ),
                timeout=aiohttp.ClientTimeout(total=5),
                headers=headers
            )
            return None if response.ok else f"HTTP Status {response.status}"
        except Exception as ex:
            return str(ex)
    
    async def async_get_available_models(self) -> List[str]:
        headers = {}
        session = async_get_clientsession(self.hass)
        async with session.get(
             ALlmBaseBackend._format_url(
                hostname=self.client_options.get(CONF_LLM_HOST),
                port=self.client_options.get(CONF_LLM_PORT),
                ssl=self.client_options.get(CONF_LLM_SSL),
                path=f"/api/tags"
            ),
            timeout=aiohttp.ClientTimeout(total=5),
            headers=headers
        ) as response:
            response.raise_for_status()
            models_result = await response.json()

        return [x["name"] for x in models_result["models"] if "embed" not in x["name"].lower()]

    async def async_send_chat_request(self, config_subentry: dict, messages: List[Dict[str, str]], tools: List[Dict]) -> AsyncGenerator[str, None]:
        """Send a chat request to Ollama and stream responses."""
        session = async_get_clientsession(self.hass)
        url = ALlmBaseBackend._format_url(
            hostname=self.client_options.get(CONF_LLM_HOST),
            port=self.client_options.get(CONF_LLM_PORT),
            ssl=self.client_options.get(CONF_LLM_SSL),
            path="/api/chat"
        )

        payload = {
            "model": config_subentry[CONF_LLM_MODEL],
            "messages": messages,
            "stream": True,
            "temperature": config_subentry[CONF_TEMPERATURE],
            "num_predict": config_subentry[CONF_MAX_TOKENS],
        }

        if tools and len(tools) > 0:
            payload["tools"] = tools
            _logger.info("Added %d tools to Ollama request", len(tools))
        
        try:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=None, sock_connect=30)) as response:
                response.raise_for_status()
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            
                            # Handle text content
                            if "message" in data and "content" in data["message"]:
                                content = data["message"]["content"]
                                if content:
                                    yield content
                            
                            # Handle native tool calls from Ollama
                            if "message" in data and "tool_calls" in data["message"]:
                                tool_calls = data["message"]["tool_calls"]
                                if tool_calls:
                                    _logger.debug("Received %d tool calls from Ollama", len(tool_calls))
                                    # Convert to our ```homeassistant block format for existing parser
                                    for tc in tool_calls:
                                        if "function" in tc:
                                            func = tc["function"]
                                            tool_json = {
                                                "tool": func.get("name", "unknown"),
                                                "arguments": func.get("arguments", {})
                                            }
                                            # Yield in format that _parse_tool_calls expects
                                            yield f"\n```homeassistant\n{json.dumps(tool_json)}\n```\n"
                                            
                        except json.JSONDecodeError:
                            _logger.debug("Failed to parse Ollama response: %s", line)
                            continue
        except Exception as err:
            _logger.error("Error calling Ollama API: %s", err, exc_info=True)
            raise