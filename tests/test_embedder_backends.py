from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

from django.template import response
import pytest

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import tests

from custom_components.ha_ragent.src.const import (
    CONF_EMBEDDING_HOST,
    CONF_EMBEDDING_MODEL,
    CONF_EMBEDDING_PORT,
    CONF_EMBEDDING_SSL,
)
from custom_components.ha_ragent.src.models.model_info import ModelInfo
from custom_components.ha_ragent.src.models.tool_embedding import LlmToolEmbedding
from custom_components.ha_ragent.src.backends.embedder.base_backend import ABaseEmbedder
from custom_components.ha_ragent.src.backends.homeassistant_mock import MockHomeAssistant


from tests.mocks import (
    MOCK_LLM_TOOLS,
    MOCK_LLM_TOOLS_EMBEDDING_OVERFLOW,
    MOCK_MESSAGES,
    MOCK_OLLAMA_EMBEDDING_CONFIG,
    MOCK_OLLAMA_EMBEDDING_CONFIG_INVALID,
    MOCK_OLLAMA_EMBEDDING_CONNECTION_USER_INPUT,
    MOCK_OLLAMA_EMBEDDING_CONNECTION_USER_INPUT_INVALID,
    MOCK_OPENAI_EMBEDDING_CONFIG,
    MOCK_OPENAI_EMBEDDING_CONFIG_INVALID,
    MOCK_OPENAI_EMBEDDING_CONNECTION_USER_INPUT,
    MOCK_OPENAI_EMBEDDING_CONNECTION_USER_INPUT_INVALID,
)
from custom_components.ha_ragent.src.backends.embedder.openai_backend import OpenAiEmbedder
from custom_components.ha_ragent.src.backends.embedder.ollama_backend import OllamaEmbedder

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

def test_init(backend_class: type[ABaseEmbedder], connection_input: dict[str, Any]) -> None:
    """Test that the backend can be initialized with a mocked Home Assistant."""
    backend = backend_class(hass, connection_input)
    assert backend._url_base == {
        "hostname": connection_input[CONF_EMBEDDING_HOST],
        "port": connection_input[CONF_EMBEDDING_PORT],
        "ssl": connection_input[CONF_EMBEDDING_SSL],
    }
    return backend

def test_url_format(backend: ABaseEmbedder, connection_input: dict[str, Any]) -> None:
    """Test that the backend can format URLs correctly."""
    url = backend.format_url(
        hostname=connection_input[CONF_EMBEDDING_HOST],
        port=connection_input[CONF_EMBEDDING_PORT],
        ssl=connection_input[CONF_EMBEDDING_SSL],
        path="/v1",
    )
    expected_url = f"{'https' if connection_input[CONF_EMBEDDING_SSL] else 'http'}://{connection_input[CONF_EMBEDDING_HOST]}{':' + str(connection_input[CONF_EMBEDDING_PORT]) if connection_input[CONF_EMBEDDING_PORT] else ''}/v1"
    assert url == expected_url

# These are reusable helpers invoked by test_all_llm_backends, not standalone tests.
test_init.__test__ = False
test_url_format.__test__ = False

async def _async_test_connection_success(backend: ABaseEmbedder, connection_input: dict[str, Any]) -> None:
    """Test that the backend can validate a connection with a mocked Home Assistant."""
    validation_error = await backend.async_validate_connection(hass, connection_input)
    assert validation_error is None, validation_error

async def _async_test_connection_failure(backend: ABaseEmbedder, connection_input_invalid: dict[str, Any]) -> None:
    """Test that the backend can handle connection failures with a mocked Home Assistant."""
    validation_error = await backend.async_validate_connection(hass, connection_input_invalid)
    assert validation_error is not None, "Expected a validation error for invalid input"

async def async_test_validate_connection(
        backend_valid: ABaseEmbedder, 
        backend_invalid: ABaseEmbedder,
        user_input_valid: dict[str, Any],
        user_input_invalid: dict[str, Any]
        ) -> None:
    """Test that the backend can validate a connection with a mocked Home Assistant."""
    await _async_test_connection_success(backend_valid, user_input_valid)
    await _async_test_connection_failure(backend_invalid, user_input_invalid)

async def _async_test_get_available_models_success(backend: ABaseEmbedder) -> None:
    """Test that the backend can retrieve available models with a mocked Home Assistant."""
    models = await backend.async_get_available_models()
    assert isinstance(models, list), "Available models should be a list"
    assert len(models) > 0, "Expected at least one available model"

async def _async_test_get_available_models_failure(backend: ABaseEmbedder) -> None:
    """Test that the backend can handle available models retrieval failures with a mocked Home Assistant."""
    try:
        await backend.async_get_available_models()
        assert False, "Expected an exception for unavailable models"
    except Exception as e:
        assert isinstance(e, Exception), f"Expected an exception, got {type(e)}"

async def async_test_get_available_models(backend_valid: ABaseEmbedder, backend_invalid: ABaseEmbedder) -> None:
    """Test that the backend can retrieve available models with a mocked Home Assistant."""
    await _async_test_get_available_models_success(backend_valid)
    await _async_test_get_available_models_failure(backend_invalid)

async def _async_test_get_model_info_success(backend: ABaseEmbedder, embedding_config: dict[str, Any]) -> None:
    """Test that the backend can retrieve model info with a mocked Home Assistant."""
    model_info = await backend.async_get_model_info(embedding_config[CONF_EMBEDDING_MODEL])
    assert isinstance(model_info, ModelInfo), "Model info should be a ModelInfo instance"
    assert model_info.name == embedding_config[CONF_EMBEDDING_MODEL], "Model name should match the requested model"
    assert model_info.context_size is None or model_info.context_size > 0, "Context size should be None or greater than 0"
    assert model_info.is_embedding_model is None or model_info.is_embedding_model in [True, False], "is_embedding_model should be None or a boolean"
    assert model_info.is_tool_model is None or model_info.is_tool_model in [True, False], "is_tool_model should be None or a boolean"

async def _async_test_get_model_info_failure(backend: ABaseEmbedder, embedding_config_invalid: dict[str, Any]) -> None:
    """Test that the backend can handle model info retrieval failures with a mocked Home Assistant."""
    try:
        await backend.async_get_model_info(embedding_config_invalid[CONF_EMBEDDING_MODEL])
        assert False, "Expected an exception for non-existent model"
    except Exception as e:
        assert isinstance(e, Exception), f"Expected an exception, got {type(e)}"

async def async_test_get_model_info(
        backend_valid: ABaseEmbedder, 
        backend_invalid: ABaseEmbedder, 
        embedding_config: dict[str, Any], 
        embedding_config_invalid: dict[str, Any]) -> None:
    """Test that the backend can retrieve model info with a mocked Home Assistant."""
    await _async_test_get_model_info_success(backend_valid, embedding_config)
    await _async_test_get_model_info_failure(backend_invalid, embedding_config_invalid)

async def async_test_preload_and_unload_model(backend: ABaseEmbedder, embedding_config: dict[str, Any]) -> None:
    """Test that the backend can preload and unload a model with a mocked Home Assistant."""
    await backend.async_preload_model(embedding_config)
    await backend.async_unload_model(embedding_config)

async def _async_test_embed_tools(backend: ABaseEmbedder, embedding_config: dict[str, Any]) -> None:
    """Test that the backend can embed tools with a mocked Home Assistant."""
    embeddings = await backend.async_embed_object(embedding_config, MOCK_LLM_TOOLS)
    assert isinstance(embeddings, list), "Response should be a list"
    assert len(embeddings) > 0, "Expected at least one response item"
    assert all(isinstance(embedding, LlmToolEmbedding) for embedding in embeddings), "Each embedding should be a ToolEmbedding instance"
    assert len(embeddings) == len(MOCK_LLM_TOOLS)

async def _async_test_embed_tools_overflow(backend: ABaseEmbedder, embedding_config: dict[str, Any]) -> None:
    """Test that the backend can handle embedding tools with a mocked Home Assistant."""
    embeddings = await backend.async_embed_object(embedding_config, MOCK_LLM_TOOLS_EMBEDDING_OVERFLOW)
    assert isinstance(embeddings, list), "Response should be a list"
    assert len(embeddings) > 0, "Expected at least one response item"
    assert all(isinstance(embedding, LlmToolEmbedding) for embedding in embeddings), "Each embedding should be a ToolEmbedding instance"
    assert len(embeddings) == len(MOCK_LLM_TOOLS_EMBEDDING_OVERFLOW)

async def _async_test_embed_tools_connection_failure(backend: ABaseEmbedder, embedding_config_invalid: dict[str, Any]) -> None:
    """Test that the backend can handle embedding tools failures with a mocked Home Assistant."""
    try:
        await backend.async_embed_object(embedding_config_invalid, MOCK_LLM_TOOLS)
        assert False, "Expected an exception for invalid embedding config"
    except Exception as e:
        assert isinstance(e, Exception), f"Expected an exception, got {type(e)}"

async def async_test_embed_tools(
        backend_valid: ABaseEmbedder, 
        backend_invalid: ABaseEmbedder,
        embedding_config: dict[str, Any],
        embedding_config_invalid: dict[str, Any]
        ) -> None:
    """Test that the backend can send a chat request with a mocked Home Assistant."""
    await _async_test_embed_tools(backend_valid, embedding_config)
    await _async_test_embed_tools_overflow(backend_valid, embedding_config)
    await _async_test_embed_tools_connection_failure(backend_invalid, embedding_config_invalid)

async def async_run_backend_test_sequence(
    backend_class: type[ABaseEmbedder],
    user_input: dict[str, Any],
    user_input_invalid: dict[str, Any],
    embedding_config: dict[str, Any],
    embedding_config_invalid: dict[str, Any]
) -> None:
    """Run every shared LLM backend helper for each supported backend."""
    backend_valid = backend_class(hass, user_input)
    backend_invalid = backend_class(hass, user_input_invalid)

    test_init(backend_class, user_input)
    test_url_format(backend_valid, user_input)
    await async_test_validate_connection(backend_valid, backend_invalid, user_input, user_input_invalid)
    await async_test_get_available_models(backend_valid, backend_invalid)
    await async_test_get_model_info(backend_valid, backend_invalid, embedding_config, embedding_config_invalid)
    await async_test_preload_and_unload_model(backend_valid, embedding_config)
    await async_test_embed_tools(backend_valid, backend_invalid, embedding_config, embedding_config_invalid)

@pytest.mark.parametrize(
    "backend_class, user_input, user_input_invalid, embedding_config, embedding_config_invalid",
    [
        pytest.param(
            OpenAiEmbedder,        
            MOCK_OPENAI_EMBEDDING_CONNECTION_USER_INPUT,
            MOCK_OPENAI_EMBEDDING_CONNECTION_USER_INPUT_INVALID,
            MOCK_OPENAI_EMBEDDING_CONFIG,
            MOCK_OPENAI_EMBEDDING_CONFIG_INVALID,
            id="OpenAiEmbedder",
        ),
        pytest.param(
            OllamaEmbedder,
            MOCK_OLLAMA_EMBEDDING_CONNECTION_USER_INPUT,
            MOCK_OLLAMA_EMBEDDING_CONNECTION_USER_INPUT_INVALID,
            MOCK_OLLAMA_EMBEDDING_CONFIG,
            MOCK_OLLAMA_EMBEDDING_CONFIG_INVALID,
            id="OllamaEmbedder",
        ),
    ],
)
def test_all_embedder_backends(
    backend_class: type[ABaseEmbedder],
    user_input: dict[str, Any],
    user_input_invalid: dict[str, Any],
    embedding_config: dict[str, Any],
    embedding_config_invalid: dict[str, Any]
) -> None:
    """Run every shared LLM backend helper for each supported backend."""
    asyncio.run(async_run_backend_test_sequence(
        backend_class,
        user_input,
        user_input_invalid,
        embedding_config,
        embedding_config_invalid
    ))