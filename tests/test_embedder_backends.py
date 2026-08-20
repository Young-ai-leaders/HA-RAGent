from __future__ import annotations

import asyncio
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
    CONF_EMBEDDING_HOST,
    CONF_EMBEDDING_MODEL,
    CONF_EMBEDDING_PORT,
    CONF_EMBEDDING_SSL,
)
from custom_components.ha_ragent.src.models.model_info import ModelInfo
from custom_components.ha_ragent.src.models.tool_embedding import LlmToolEmbedding
from custom_components.ha_ragent.src.backends.embedder.base_backend import ABaseEmbedder
from custom_components.ha_ragent.src.mock import MockHomeAssistant


from tests.mocks import (
    MOCK_LLM_TOOLS,
    MOCK_LLM_TOOLS_EMBEDDING_OVERFLOW,
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

@dataclass(frozen=True)
class EmbedderCase:
    """Configuration for one embedder backend test case."""

    backend_class: type[ABaseEmbedder]
    user_input: dict[str, Any]
    user_input_invalid: dict[str, Any]
    embedding_config: dict[str, Any]
    embedding_config_invalid: dict[str, Any]

EMBEDDER_BACKENDS = [
    EmbedderCase(
        OpenAiEmbedder,
        MOCK_OPENAI_EMBEDDING_CONNECTION_USER_INPUT,
        MOCK_OPENAI_EMBEDDING_CONNECTION_USER_INPUT_INVALID,
        MOCK_OPENAI_EMBEDDING_CONFIG,
        MOCK_OPENAI_EMBEDDING_CONFIG_INVALID,
    ),
    EmbedderCase(
        OllamaEmbedder,
        MOCK_OLLAMA_EMBEDDING_CONNECTION_USER_INPUT,
        MOCK_OLLAMA_EMBEDDING_CONNECTION_USER_INPUT_INVALID,
        MOCK_OLLAMA_EMBEDDING_CONFIG,
        MOCK_OLLAMA_EMBEDDING_CONFIG_INVALID,
    ),
]

@pytest.fixture(params=EMBEDDER_BACKENDS, ids=lambda case: case.backend_class.__name__)
def embedder_case(request: pytest.FixtureRequest) -> EmbedderCase:
    """Provide every supported embedder from the central backend list."""
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

async def _async_test_connection_success(backend: ABaseEmbedder, connection_input: dict[str, Any]) -> None:
    """Test that the backend can validate a connection with a mocked Home Assistant."""
    validation_error = await backend.async_validate_connection(backend._hass, connection_input)
    assert validation_error is None, validation_error

async def _async_test_connection_failure(backend: ABaseEmbedder, connection_input_invalid: dict[str, Any]) -> None:
    """Test that the backend can handle connection failures with a mocked Home Assistant."""
    validation_error = await backend.async_validate_connection(backend._hass, connection_input_invalid)
    assert validation_error is not None, "Expected a validation error for invalid input"

async def _async_test_validate_connection(
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
    with pytest.raises(Exception):
        await backend.async_get_available_models()

async def _async_test_get_available_models(backend_valid: ABaseEmbedder, backend_invalid: ABaseEmbedder) -> None:
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
    with pytest.raises(Exception):
        await backend.async_get_model_info(embedding_config_invalid[CONF_EMBEDDING_MODEL])

async def _async_test_get_model_info(
        backend_valid: ABaseEmbedder, 
        backend_invalid: ABaseEmbedder, 
        embedding_config: dict[str, Any], 
        embedding_config_invalid: dict[str, Any]) -> None:
    """Test that the backend can retrieve model info with a mocked Home Assistant."""
    await _async_test_get_model_info_success(backend_valid, embedding_config)
    await _async_test_get_model_info_failure(backend_invalid, embedding_config_invalid)

async def _async_test_preload_and_unload_model(backend: ABaseEmbedder, embedding_config: dict[str, Any]) -> None:
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
    with pytest.raises(Exception):
        await backend.async_embed_object(embedding_config_invalid, MOCK_LLM_TOOLS)

async def _async_test_embed_tool_scenarios(
        backend_valid: ABaseEmbedder, 
        backend_invalid: ABaseEmbedder,
        embedding_config: dict[str, Any],
        embedding_config_invalid: dict[str, Any]
        ) -> None:
    """Test that the backend can send a chat request with a mocked Home Assistant."""
    await _async_test_embed_tools(backend_valid, embedding_config)
    await _async_test_embed_tools_overflow(backend_valid, embedding_config)
    await _async_test_embed_tools_connection_failure(backend_invalid, embedding_config_invalid)

def test_init(embedder_case: EmbedderCase, hass: MockHomeAssistant) -> None:
    """Test that the backend can be initialized with a mocked Home Assistant."""
    backend = embedder_case.backend_class(hass, embedder_case.user_input)
    assert backend._url_base == {
        "hostname": embedder_case.user_input[CONF_EMBEDDING_HOST],
        "port": embedder_case.user_input[CONF_EMBEDDING_PORT],
        "ssl": embedder_case.user_input[CONF_EMBEDDING_SSL],
    }

def test_url_format(embedder_case: EmbedderCase, hass: MockHomeAssistant) -> None:
    """Test that the backend can format URLs correctly."""
    backend = embedder_case.backend_class(hass, embedder_case.user_input)
    connection_input = embedder_case.user_input
    url = backend.format_url(
        hostname=connection_input[CONF_EMBEDDING_HOST],
        port=connection_input[CONF_EMBEDDING_PORT],
        ssl=connection_input[CONF_EMBEDDING_SSL],
        path="/v1",
    )
    expected_url = f"{'https' if connection_input[CONF_EMBEDDING_SSL] else 'http'}://{connection_input[CONF_EMBEDDING_HOST]}{':' + str(connection_input[CONF_EMBEDDING_PORT]) if connection_input[CONF_EMBEDDING_PORT] else ''}/v1"
    assert url == expected_url

def test_validate_connection(embedder_case: EmbedderCase, hass: MockHomeAssistant) -> None:
    """Test connection validation for every embedder backend."""
    backend_valid = embedder_case.backend_class(hass, embedder_case.user_input)
    backend_invalid = embedder_case.backend_class(hass, embedder_case.user_input_invalid)
    _run_async_test(hass,
        _async_test_validate_connection(
            backend_valid,
            backend_invalid,
            embedder_case.user_input,
            embedder_case.user_input_invalid,
        )
    )

def test_get_available_models(embedder_case: EmbedderCase, hass: MockHomeAssistant) -> None:
    """Test model discovery for every embedder backend."""
    backend_valid = embedder_case.backend_class(hass, embedder_case.user_input)
    backend_invalid = embedder_case.backend_class(hass, embedder_case.user_input_invalid)
    _run_async_test(hass, _async_test_get_available_models(backend_valid, backend_invalid))

def test_get_model_info(embedder_case: EmbedderCase, hass: MockHomeAssistant) -> None:
    """Test model information retrieval for every embedder backend."""
    backend_valid = embedder_case.backend_class(hass, embedder_case.user_input)
    backend_invalid = embedder_case.backend_class(hass, embedder_case.user_input_invalid)
    _run_async_test(hass,
        _async_test_get_model_info(
            backend_valid,
            backend_invalid,
            embedder_case.embedding_config,
            embedder_case.embedding_config_invalid,
        )
    )

def test_preload_and_unload_model(embedder_case: EmbedderCase, hass: MockHomeAssistant) -> None:
    """Test model lifecycle operations for every embedder backend."""
    backend = embedder_case.backend_class(hass, embedder_case.user_input)
    _run_async_test(hass,
        _async_test_preload_and_unload_model(
            backend,
            embedder_case.embedding_config,
        )
    )

def test_embed_tools(embedder_case: EmbedderCase, hass: MockHomeAssistant) -> None:
    """Test tool embedding for every embedder backend."""
    backend_valid = embedder_case.backend_class(hass, embedder_case.user_input)
    backend_invalid = embedder_case.backend_class(hass, embedder_case.user_input_invalid)
    _run_async_test(hass,
        _async_test_embed_tool_scenarios(
            backend_valid,
            backend_invalid,
            embedder_case.embedding_config,
            embedder_case.embedding_config_invalid,
        )
    )
