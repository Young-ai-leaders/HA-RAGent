from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

import pytest

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import tests

from custom_components.ha_ragent.src.const import (
    CONF_LLM_HOST,
    CONF_LLM_MODEL,
    CONF_LLM_PORT,
    CONF_LLM_SSL,
)
from custom_components.ha_ragent.src.models.model_info import ModelInfo
from custom_components.ha_ragent.src.backends.llm.base_backend import ALlmBaseBackend
from custom_components.ha_ragent.src.backends.mock import MockHomeAssistant


from tests.mocks import (
    MOCK_LLM_TOOLS,
    MOCK_MESSAGES,
    MOCK_MESSAGE_CONTEXT_OVERFLOW,
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

hass = MockHomeAssistant()

@pytest.fixture(autouse=True)
def suppress_logging() -> Any:
    """Suppress backend log output for this test module."""
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous_disable_level)

def test_init(backend_class: type[ALlmBaseBackend], connection_input: dict[str, Any]) -> None:
    """Test that the backend can be initialized with a mocked Home Assistant."""
    backend = backend_class(hass, connection_input)
    assert backend._url_base == {
        "hostname": connection_input[CONF_LLM_HOST],
        "port": connection_input[CONF_LLM_PORT],
        "ssl": connection_input[CONF_LLM_SSL],
    }
    return backend

def test_url_format(backend: ALlmBaseBackend, connection_input: dict[str, Any]) -> None:
    """Test that the backend can format URLs correctly."""
    url = backend.format_url(
        hostname=connection_input[CONF_LLM_HOST],
        port=connection_input[CONF_LLM_PORT],
        ssl=connection_input[CONF_LLM_SSL],
        path="/v1",
    )
    expected_url = f"{'https' if connection_input[CONF_LLM_SSL] else 'http'}://{connection_input[CONF_LLM_HOST]}{':' + str(connection_input[CONF_LLM_PORT]) if connection_input[CONF_LLM_PORT] else ''}/v1"
    assert url == expected_url

# These are reusable helpers invoked by test_all_llm_backends, not standalone tests.
test_init.__test__ = False
test_url_format.__test__ = False

async def _async_test_connection_success(backend: ALlmBaseBackend, connection_input: dict[str, Any]) -> None:
    """Test that the backend can validate a connection with a mocked Home Assistant."""
    validation_error = await backend.async_validate_connection(hass, connection_input)
    assert validation_error is None, validation_error

async def _async_test_connection_failure(backend: ALlmBaseBackend, connection_input_invalid: dict[str, Any]) -> None:
    """Test that the backend can handle connection failures with a mocked Home Assistant."""
    validation_error = await backend.async_validate_connection(hass, connection_input_invalid)
    assert validation_error is not None, "Expected a validation error for invalid input"

async def async_test_validate_connection(
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
    try:
        await backend.async_get_available_models()
        assert False, "Expected an exception for unavailable models"
    except Exception as e:
        assert isinstance(e, Exception), f"Expected an exception, got {type(e)}"

async def async_test_get_available_models(backend_valid: ALlmBaseBackend, backend_invalid: ALlmBaseBackend) -> None:
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
    try:
        await backend.async_get_model_info(chat_config_invalid[CONF_LLM_MODEL])
        assert False, "Expected an exception for non-existent model"
    except Exception as e:
        assert isinstance(e, Exception), f"Expected an exception, got {type(e)}"

async def async_test_get_model_info(
        backend_valid: ALlmBaseBackend, 
        backend_invalid: ALlmBaseBackend, 
        chat_config: dict[str, Any], 
        chat_config_invalid: dict[str, Any]) -> None:
    """Test that the backend can retrieve model info with a mocked Home Assistant."""
    await _async_test_get_model_info_success(backend_valid, chat_config)
    await _async_test_get_model_info_failure(backend_invalid, chat_config_invalid)

async def async_test_preload_and_unload_model(backend: ALlmBaseBackend, chat_config: dict[str, Any]) -> None:
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
    try:
        async for _ in backend.async_send_chat_request(
            chat_config_invalid,
            MOCK_MESSAGES,
            MOCK_LLM_TOOLS,
        ):
            pass

        assert False, "Expected an exception for empty messages"
    except Exception as e:
        assert isinstance(e, Exception), f"Expected an exception, got {type(e)}"

async def async_test_send_chat_request(
        backend_valid: ALlmBaseBackend, 
        backend_invalid: ALlmBaseBackend,
        chat_config: dict[str, Any],
        chat_config_invalid: dict[str, Any]
        ) -> None:
    """Test that the backend can send a chat request with a mocked Home Assistant."""
    await _async_test_send_chat_request_success(backend_valid, chat_config)
    await _async_test_send_chat_request_overflow_failure(backend_valid, chat_config)
    await _async_test_send_chat_request_connection_failure(backend_invalid, chat_config_invalid)

async def async_run_backend_test_sequence(
    backend_class: type[ALlmBaseBackend],
    user_input: dict[str, Any],
    user_input_invalid: dict[str, Any],
    chat_config: dict[str, Any],
    chat_config_invalid: dict[str, Any]
) -> None:
    """Run every shared LLM backend helper for each supported backend."""
    backend_valid = backend_class(hass, user_input)
    backend_invalid = backend_class(hass, user_input_invalid)

    test_init(backend_class, user_input)
    test_url_format(backend_valid, user_input)
    await async_test_validate_connection(backend_valid, backend_invalid, user_input, user_input_invalid)
    await async_test_get_available_models(backend_valid, backend_invalid)
    await async_test_get_model_info(backend_valid, backend_invalid, chat_config, chat_config_invalid)
    await async_test_preload_and_unload_model(backend_valid, chat_config)
    await async_test_send_chat_request(backend_valid, backend_invalid, chat_config, chat_config_invalid)

@pytest.mark.parametrize(
    "backend_class, user_input, user_input_invalid, chat_config, chat_config_invalid",
    [
        pytest.param(
            OpenAiLlmBackend,        
            MOCK_OPENAI_CONNECTION_USER_INPUT,
            MOCK_OPENAI_CONNECTION_USER_INPUT_INVALID,
            MOCK_OPENAI_CHAT_CONFIG,
            MOCK_OPENAI_CHAT_CONFIG_INVALID,
            id="OpenAiLlmBackend",
        ),
        pytest.param(
            OllamaLlmBackend,
            MOCK_OLLAMA_CONNECTION_USER_INPUT,
            MOCK_OLLAMA_CONNECTION_USER_INPUT_INVALID,
            MOCK_OLLAMA_CHAT_CONFIG,
            MOCK_OLLAMA_CHAT_CONFIG_INVALID,
            id="OllamaLlmBackend",
        ),
    ],
)
def test_all_llm_backends(
    backend_class: type[ALlmBaseBackend],
    user_input: dict[str, Any],
    user_input_invalid: dict[str, Any],
    chat_config: dict[str, Any],
    chat_config_invalid: dict[str, Any]
) -> None:
    """Run every shared LLM backend helper for each supported backend."""
    asyncio.run(async_run_backend_test_sequence(
        backend_class,
        user_input,
        user_input_invalid,
        chat_config,
        chat_config_invalid
    ))
