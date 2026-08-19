from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Awaitable

import pytest

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import tests

from custom_components.ha_ragent.src.const import (
    CONF_LLM_HOST,
    CONF_LLM_MODEL,
    CONF_LLM_PORT,
    CONF_LLM_SSL,
    RAGENT_SEMANTIC_SEARCH_TOOL_NAME,
)
from custom_components.ha_ragent.src.models.model_info import ModelInfo
from custom_components.ha_ragent.src.models.tool import LlmTool
from custom_components.ha_ragent.src.models.chat_message import ChatMessage
from custom_components.ha_ragent.src.backends.llm.base_backend import ALlmBaseBackend
from custom_components.ha_ragent.src.backends.mock import MockHomeAssistant


from tests.mocks import (
    MOCK_LLM_TOOLS,
    MOCK_MESSAGES,
    MOCK_MESSAGE_CONTEXT_OVERFLOW,
    MOCK_TOOL_HISTORY,
    MOCK_OLLAMA_CHAT_CONFIG,
    MOCK_OLLAMA_CHAT_CONFIG_INVALID,
    MOCK_OLLAMA_CONNECTION_USER_INPUT,
    MOCK_OLLAMA_CONNECTION_USER_INPUT_INVALID,
    MOCK_OPENAI_CHAT_CONFIG,
    MOCK_OPENAI_CHAT_CONFIG_INVALID,
    MOCK_OPENAI_CONNECTION_USER_INPUT,
    MOCK_OPENAI_CONNECTION_USER_INPUT_INVALID,
)
from custom_components.ha_ragent.src.backends.llm.openai_backend import OpenAiLlmBackend
from custom_components.ha_ragent.src.backends.llm.ollama_backend import OllamaLlmBackend

@dataclass(frozen=True)
class BackendCase:
    """Configuration for one LLM backend test case."""
    backend_class: type[ALlmBaseBackend]
    user_input: dict[str, Any]
    user_input_invalid: dict[str, Any]
    chat_config: dict[str, Any]
    chat_config_invalid: dict[str, Any]

LLM_BACKENDS = [
    BackendCase(
        OpenAiLlmBackend,
        MOCK_OPENAI_CONNECTION_USER_INPUT,
        MOCK_OPENAI_CONNECTION_USER_INPUT_INVALID,
        MOCK_OPENAI_CHAT_CONFIG,
        MOCK_OPENAI_CHAT_CONFIG_INVALID,
    ),
    BackendCase(
        OllamaLlmBackend,
        MOCK_OLLAMA_CONNECTION_USER_INPUT,
        MOCK_OLLAMA_CONNECTION_USER_INPUT_INVALID,
        MOCK_OLLAMA_CHAT_CONFIG,
        MOCK_OLLAMA_CHAT_CONFIG_INVALID,
    ),
]

@pytest.fixture(params=LLM_BACKENDS, ids=lambda case: case.backend_class.__name__)
def backend_case(request: pytest.FixtureRequest) -> BackendCase:
    """Provide every supported backend from the central backend list."""
    return request.param

@pytest.fixture
def hass() -> MockHomeAssistant:
    """Provide an isolated Home Assistant mock for each test."""
    return MockHomeAssistant()

def _run_async_test(hass: MockHomeAssistant, test: Awaitable[None]) -> None:
    """Run an async test and close resources before its event loop exits."""
    async def run() -> None:
        try:
            await test
        finally:
            await hass.async_close()

    asyncio.run(run())

@pytest.fixture(autouse=True)
def suppress_logging() -> Any:
    """Suppress backend log output for this test module."""
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous_disable_level)

async def _async_test_connection_success(backend: ALlmBaseBackend, connection_input: dict[str, Any]) -> None:
    """Test that the backend can validate a connection with a mocked Home Assistant."""
    validation_error = await backend.async_validate_connection(backend._hass, connection_input)
    assert validation_error is None, validation_error

async def _async_test_connection_failure(backend: ALlmBaseBackend, connection_input_invalid: dict[str, Any]) -> None:
    """Test that the backend can handle connection failures with a mocked Home Assistant."""
    validation_error = await backend.async_validate_connection(backend._hass, connection_input_invalid)
    assert validation_error is not None, "Expected a validation error for invalid input"

async def _async_test_validate_connection(
        backend_valid: ALlmBaseBackend, 
        backend_invalid: ALlmBaseBackend,
        user_input_valid: dict[str, Any],
        user_input_invalid: dict[str, Any]
        ) -> None:
    """Test that the backend can validate a connection with a mocked Home Assistant."""
    await _async_test_connection_success(backend_valid, user_input_valid)
    await _async_test_connection_failure(backend_invalid, user_input_invalid)

async def _async_test_get_available_models_success(backend: ALlmBaseBackend) -> None:
    """Test that the backend can retrieve available models with a mocked Home Assistant."""
    models = await backend.async_get_available_models()
    assert isinstance(models, list), "Available models should be a list"
    assert len(models) > 0, "Expected at least one available model"

async def _async_test_get_available_models_failure(backend: ALlmBaseBackend) -> None:
    """Test that the backend can handle available models retrieval failures with a mocked Home Assistant."""
    with pytest.raises(Exception):
        await backend.async_get_available_models()

async def _async_test_get_available_models(backend_valid: ALlmBaseBackend, backend_invalid: ALlmBaseBackend) -> None:
    """Test that the backend can retrieve available models with a mocked Home Assistant."""
    await _async_test_get_available_models_success(backend_valid)
    await _async_test_get_available_models_failure(backend_invalid)

async def _async_test_get_model_info_success(backend: ALlmBaseBackend, chat_config: dict[str, Any]) -> None:
    """Test that the backend can retrieve model info with a mocked Home Assistant."""
    model_info = await backend.async_get_model_info(chat_config[CONF_LLM_MODEL])
    assert isinstance(model_info, ModelInfo), "Model info should be a ModelInfo instance"
    assert model_info.name == chat_config[CONF_LLM_MODEL], "Model name should match the requested model"
    assert model_info.context_size is None or model_info.context_size > 0, "Context size should be None or greater than 0"
    assert model_info.is_embedding_model is None or model_info.is_embedding_model in [True, False], "is_embedding_model should be None or a boolean"
    assert model_info.is_tool_model is None or model_info.is_tool_model in [True, False], "is_tool_model should be None or a boolean"

async def _async_test_get_model_info_failure(backend: ALlmBaseBackend, chat_config_invalid: dict[str, Any]) -> None:
    """Test that the backend can handle model info retrieval failures with a mocked Home Assistant."""
    with pytest.raises(Exception):
        await backend.async_get_model_info(chat_config_invalid[CONF_LLM_MODEL])

async def _async_test_get_model_info(
        backend_valid: ALlmBaseBackend, 
        backend_invalid: ALlmBaseBackend, 
        chat_config: dict[str, Any], 
        chat_config_invalid: dict[str, Any]) -> None:
    """Test that the backend can retrieve model info with a mocked Home Assistant."""
    await _async_test_get_model_info_success(backend_valid, chat_config)
    await _async_test_get_model_info_failure(backend_invalid, chat_config_invalid)

async def _async_test_preload_and_unload_model(backend: ALlmBaseBackend, chat_config: dict[str, Any]) -> None:
    """Test that the backend can preload and unload a model with a mocked Home Assistant."""
    await backend.async_preload_model(chat_config)
    await backend.async_unload_model(chat_config)

async def _async_test_send_chat_request_success(backend: ALlmBaseBackend, chat_config: dict[str, Any]) -> None:
    """Test that the backend can send a chat request with a mocked Home Assistant."""
    response = [
        item
        async for item in backend.async_send_chat_request(
            chat_config,
            MOCK_MESSAGES,
            MOCK_LLM_TOOLS,
        )
    ]

    assert isinstance(response, list), "Response should be a list"
    assert len(response) > 0, "Expected at least one response item"

async def _async_test_send_chat_request_overflow_failure(backend: ALlmBaseBackend, chat_config: dict[str, Any]) -> None:
    """Test handling chat request failures caused by message overflow."""
    response = [
        item
        async for item in backend.async_send_chat_request(
            chat_config,
            MOCK_MESSAGE_CONTEXT_OVERFLOW,
            MOCK_LLM_TOOLS,
        )
    ]

    assert isinstance(response, list), "Response should be a list"
    assert len(response) > 0, "Expected at least one response item"

async def _async_test_send_chat_request_connection_failure(backend: ALlmBaseBackend, chat_config_invalid: dict[str, Any]) -> None:
    """Test handling chat request failures caused by empty messages."""
    with pytest.raises(Exception):
        async for _ in backend.async_send_chat_request(
            chat_config_invalid,
            MOCK_MESSAGES,
            MOCK_LLM_TOOLS,
        ):
            pass

async def _async_test_send_chat_request(
        backend_valid: ALlmBaseBackend, 
        backend_invalid: ALlmBaseBackend,
        chat_config: dict[str, Any],
        chat_config_invalid: dict[str, Any]
        ) -> None:
    """Test that the backend can send a chat request with a mocked Home Assistant."""
    await _async_test_send_chat_request_success(backend_valid, chat_config)
    await _async_test_send_chat_request_overflow_failure(backend_valid, chat_config)
    await _async_test_send_chat_request_connection_failure(backend_invalid, chat_config_invalid)

def test_init(backend_case: BackendCase, hass: MockHomeAssistant) -> None:
    """Test that the backend can be initialized with a mocked Home Assistant."""
    backend = backend_case.backend_class(hass, backend_case.user_input)
    assert backend._url_base == {
        "hostname": backend_case.user_input[CONF_LLM_HOST],
        "port": backend_case.user_input[CONF_LLM_PORT],
        "ssl": backend_case.user_input[CONF_LLM_SSL],
    }

def test_url_format(backend_case: BackendCase, hass: MockHomeAssistant) -> None:
    """Test that the backend can format URLs correctly."""
    backend = backend_case.backend_class(hass, backend_case.user_input)
    connection_input = backend_case.user_input
    url = backend.format_url(
        hostname=connection_input[CONF_LLM_HOST],
        port=connection_input[CONF_LLM_PORT],
        ssl=connection_input[CONF_LLM_SSL],
        path="/v1",
    )
    expected_url = f"{'https' if connection_input[CONF_LLM_SSL] else 'http'}://{connection_input[CONF_LLM_HOST]}{':' + str(connection_input[CONF_LLM_PORT]) if connection_input[CONF_LLM_PORT] else ''}/v1"
    assert url == expected_url


def test_tool_names_are_split_for_request_logging() -> None:
    """Required tools are listed separately from RAG-selected tools."""
    tools = [
        LlmTool(
            name="HassTurnOn",
            description="Turn on",
            parameters={},
            metadata={},
        ),
        LlmTool(
            name=RAGENT_SEMANTIC_SEARCH_TOOL_NAME,
            description="Search",
            parameters={},
            metadata={},
        ),
    ]

    tool_names, required_tool_names = ALlmBaseBackend.split_tool_names(tools)

    assert tool_names == ["HassTurnOn"]
    assert required_tool_names == [RAGENT_SEMANTIC_SEARCH_TOOL_NAME]

def test_validate_connection(backend_case: BackendCase, hass: MockHomeAssistant) -> None:
    """Test connection validation for every backend."""
    backend_valid = backend_case.backend_class(hass, backend_case.user_input)
    backend_invalid = backend_case.backend_class(hass, backend_case.user_input_invalid)
    _run_async_test(hass,
        _async_test_validate_connection(
            backend_valid,
            backend_invalid,
            backend_case.user_input,
            backend_case.user_input_invalid,
        )
    )

def test_get_available_models(backend_case: BackendCase, hass: MockHomeAssistant) -> None:
    """Test model discovery for every backend."""
    backend_valid = backend_case.backend_class(hass, backend_case.user_input)
    backend_invalid = backend_case.backend_class(hass, backend_case.user_input_invalid)
    _run_async_test(hass, _async_test_get_available_models(backend_valid, backend_invalid))

def test_get_model_info(backend_case: BackendCase, hass: MockHomeAssistant) -> None:
    """Test model information retrieval for every backend."""
    backend_valid = backend_case.backend_class(hass, backend_case.user_input)
    backend_invalid = backend_case.backend_class(hass, backend_case.user_input_invalid)
    _run_async_test(hass,
        _async_test_get_model_info(
            backend_valid,
            backend_invalid,
            backend_case.chat_config,
            backend_case.chat_config_invalid,
        )
    )

def test_preload_and_unload_model(backend_case: BackendCase, hass: MockHomeAssistant) -> None:
    """Test model lifecycle operations for every backend."""
    backend = backend_case.backend_class(hass, backend_case.user_input)
    _run_async_test(hass, _async_test_preload_and_unload_model(backend, backend_case.chat_config))

def test_prepares_linked_tool_history(backend_case: BackendCase, hass: MockHomeAssistant) -> None:
    """Test that a backend preserves the link between tool calls and results."""
    backend = backend_case.backend_class(hass, backend_case.user_input)
    messages = backend.format_messages_for_backend(MOCK_TOOL_HISTORY)
    tool_call = messages[2]["tool_calls"][0]
    tool_result = messages[3]

    arguments = tool_call["function"]["arguments"]
    if isinstance(arguments, str):
        arguments = json.loads(arguments)

    assert arguments == {"name": "Desk light"}
    assert (
        tool_result.get("tool_call_id") == tool_call.get("id")
        or tool_result.get("tool_name") == tool_call["function"]["name"]
    )

def test_openai_truncation_keeps_complete_turns() -> None:
    """Test that OpenAI truncation does not leave orphaned tool results."""
    messages = [
        *MOCK_TOOL_HISTORY,
        ChatMessage(role="user", content="What is its state now?"),
    ]
    max_chars = sum(
        len(json.dumps(message, default=str))
        for message in (messages[0], messages[-1])
    )
    truncated = OpenAiLlmBackend._truncate_messages(messages, max_chars)

    assert [message["role"] for message in truncated] == ["system", "user"]
    assert truncated[-1]["content"] == "What is its state now?"

def test_send_chat_request(backend_case: BackendCase, hass: MockHomeAssistant) -> None:
    """Test chat requests for every backend."""
    backend_valid = backend_case.backend_class(hass, backend_case.user_input)
    backend_invalid = backend_case.backend_class(hass, backend_case.user_input_invalid)
    _run_async_test(hass,
        _async_test_send_chat_request(
            backend_valid,
            backend_invalid,
            backend_case.chat_config,
            backend_case.chat_config_invalid,
        )
    )
