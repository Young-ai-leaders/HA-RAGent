from __future__ import annotations

from custom_components.ha_ragent.src.const import (
    CONF_CONTEXT_LENGTH,
    CONF_ENABLE_MODEL_THINKING,
    CONF_LLM_API_KEY,
    CONF_LLM_HOST,
    CONF_LLM_MODEL,
    CONF_LLM_PORT,
    CONF_LLM_SSL,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
)
from custom_components.ha_ragent.src.models.tool import LlmTool


MOCK_LLM_DEFAULT_OPTIONS = {
    CONF_LLM_HOST: "127.0.0.1",
    CONF_LLM_SSL: False,
    CONF_LLM_API_KEY: None,
}

MOCK_OPENAI_CONNECTION_USER_INPUT = {
    **MOCK_LLM_DEFAULT_OPTIONS,
    CONF_LLM_PORT: 8080,
}

MOCK_OPENAI_CONNECTION_USER_INPUT_INVALID = {
    **MOCK_LLM_DEFAULT_OPTIONS,
    CONF_LLM_HOST: "invalid_host",
    CONF_LLM_PORT: 8080,
}

MOCK_OLLAMA_CONNECTION_USER_INPUT = {
    **MOCK_LLM_DEFAULT_OPTIONS,
    CONF_LLM_PORT: 11434,
}

MOCK_OLLAMA_CONNECTION_USER_INPUT_INVALID = {
    **MOCK_LLM_DEFAULT_OPTIONS,
    CONF_LLM_HOST: "invalid_host",
    CONF_LLM_PORT: 11434,
}

MOCK_OPENAI_CHAT_CONFIG = {
    **MOCK_OPENAI_CONNECTION_USER_INPUT,
    CONF_LLM_MODEL: "ibm-granite/granite-4.1-3b-GGUF:Q4_K_M",
    CONF_TEMPERATURE: 0.2,
    CONF_MAX_TOKENS: 128,
    CONF_ENABLE_MODEL_THINKING: False,
}

MOCK_OPENAI_CHAT_CONFIG_INVALID = {
    **MOCK_OPENAI_CONNECTION_USER_INPUT,
    CONF_LLM_MODEL: "invalid_model",
    CONF_TEMPERATURE: 0.2,
    CONF_MAX_TOKENS: 128,
    CONF_ENABLE_MODEL_THINKING: False,
}

MOCK_OLLAMA_CHAT_CONFIG = {
    **MOCK_OLLAMA_CONNECTION_USER_INPUT,
    CONF_LLM_MODEL: "qwen3.5:0.8b",
    CONF_TEMPERATURE: 0.2,
    CONF_MAX_TOKENS: 128,
    CONF_ENABLE_MODEL_THINKING: False,
    CONF_CONTEXT_LENGTH: 4096,
}

MOCK_OLLAMA_CHAT_CONFIG_INVALID = {
    **MOCK_OLLAMA_CONNECTION_USER_INPUT,
    CONF_LLM_MODEL: "invalid_model",
    CONF_TEMPERATURE: 0.2,
    CONF_MAX_TOKENS: 128,
    CONF_ENABLE_MODEL_THINKING: False,
    CONF_CONTEXT_LENGTH: 4096,
}

MOCK_LLM_CHAT_CONFIG = MOCK_OPENAI_CHAT_CONFIG

MOCK_LLM_TOOLS = [
    LlmTool(
        name="test_tool",
        description="A tool used by backend tests.",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
        },
    ),
    LlmTool(
        name="another_tool",
        description="Another tool used by backend tests.",
        parameters={
            "type": "object",
            "properties": {"number": {"type": "integer"}},
        },
    ),
]

MOCK_MESSAGES = [{"role": "user", "content": "Hello"}]
MOCK_MESSAGE_CONTEXT_OVERFLOW = [{"role": "user", "content": "Message Overflow" * 10000}]
