from __future__ import annotations

from custom_components.ha_ragent.src.const import (
    CONF_CONTEXT_LENGTH,
    CONF_ENABLE_MODEL_THINKING,
    CONF_EMBEDDING_API_KEY,
    CONF_EMBEDDING_HOST,
    CONF_EMBEDDING_MODEL,
    CONF_EMBEDDING_PORT,
    CONF_EMBEDDING_SSL,
    CONF_LLM_API_KEY,
    CONF_LLM_HOST,
    CONF_LLM_MODEL,
    CONF_LLM_PORT,
    CONF_LLM_SSL,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    CONF_VECTOR_DB_HOST,
    CONF_VECTOR_DB_NAME,
    CONF_VECTOR_DB_PASSWORD,
    CONF_VECTOR_DB_PORT,
    CONF_VECTOR_DB_SSL,
    CONF_VECTOR_DB_USERNAME,
)
from custom_components.ha_ragent.src.models.chat.chat_message import (
    ChatFunction,
    ChatMessage,
    ChatToolCall,
)
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool


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
    CONF_LLM_MODEL: "Qwen/Qwen3-1.7B-GGUF:Q8_0",
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
    CONF_LLM_MODEL: "qwen3:1.7b",
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

MOCK_FAISS_DB_CONFIG = {
    CONF_VECTOR_DB_NAME: "ha_ragent_test",
}

MOCK_MONGODB_CONNECTION_USER_INPUT = {
    CONF_VECTOR_DB_HOST: "127.0.0.1",
    CONF_VECTOR_DB_PORT: 27017,
    CONF_VECTOR_DB_SSL: False,
    CONF_VECTOR_DB_USERNAME: "admin",
    CONF_VECTOR_DB_PASSWORD: "mongodb",
}

MOCK_MONGODB_CONNECTION_USER_INPUT_INVALID = {
    **MOCK_MONGODB_CONNECTION_USER_INPUT,
    CONF_VECTOR_DB_HOST: "invalid_host",
}

MOCK_CHROMADB_CONNECTION_USER_INPUT = {
    CONF_VECTOR_DB_HOST: "127.0.0.1",
    CONF_VECTOR_DB_PORT: 8000,
    CONF_VECTOR_DB_SSL: False,
}

MOCK_CHROMADB_CONNECTION_USER_INPUT_INVALID = {
    **MOCK_CHROMADB_CONNECTION_USER_INPUT,
    CONF_VECTOR_DB_HOST: "invalid_host",
}

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

MOCK_LLM_TOOLS_EMBEDDING_OVERFLOW = [
    *MOCK_LLM_TOOLS,
    LlmTool(
        name="overflow_tool",
        description="A tool with a very long description to test embedding context overflow.",
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "".join(f"Overflow text INDEX: {i}" for i in range(10000)),
                }
            },
        },
    ),
]

MOCK_EMBEDDING_DEFAULT_OPTIONS = {
    CONF_EMBEDDING_HOST: "127.0.0.1",
    CONF_EMBEDDING_SSL: False,
    CONF_EMBEDDING_API_KEY: None,
}

MOCK_OPENAI_EMBEDDING_CONNECTION_USER_INPUT = {
    **MOCK_EMBEDDING_DEFAULT_OPTIONS,
    CONF_EMBEDDING_PORT: 8081,
}

MOCK_OPENAI_EMBEDDING_CONNECTION_USER_INPUT_INVALID = {
    **MOCK_EMBEDDING_DEFAULT_OPTIONS,
    CONF_EMBEDDING_HOST: "invalid_host",
    CONF_EMBEDDING_PORT: 8081,
}

MOCK_OLLAMA_EMBEDDING_CONNECTION_USER_INPUT = {
    **MOCK_EMBEDDING_DEFAULT_OPTIONS,
    CONF_EMBEDDING_PORT: 11434,
}

MOCK_OLLAMA_EMBEDDING_CONNECTION_USER_INPUT_INVALID = {
    **MOCK_EMBEDDING_DEFAULT_OPTIONS,
    CONF_EMBEDDING_HOST: "invalid_host",
    CONF_EMBEDDING_PORT: 11434,
}

MOCK_OPENAI_EMBEDDING_CONFIG = {
    **MOCK_OPENAI_EMBEDDING_CONNECTION_USER_INPUT,
    CONF_EMBEDDING_MODEL: "nomic-ai/nomic-embed-text-v1.5-GGUF:Q4_K_M",
}

MOCK_OPENAI_EMBEDDING_CONFIG_INVALID = {
    **MOCK_OPENAI_EMBEDDING_CONNECTION_USER_INPUT,
    CONF_EMBEDDING_MODEL: "invalid_model",
}

MOCK_OLLAMA_EMBEDDING_CONFIG = {
    **MOCK_OLLAMA_EMBEDDING_CONNECTION_USER_INPUT,
    CONF_EMBEDDING_MODEL: "all-minilm:33m",
}

MOCK_OLLAMA_EMBEDDING_CONFIG_INVALID = {
    **MOCK_OLLAMA_EMBEDDING_CONNECTION_USER_INPUT,
    CONF_EMBEDDING_MODEL: "invalid_model",
}

MOCK_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello"}
]

MOCK_TOOL_HISTORY: list[ChatMessage] = [
    ChatMessage(role="system", content="Follow instructions."),
    ChatMessage(role="user", content="Turn on the desk light."),
    ChatMessage(
        role="assistant",
        content="",
        tool_calls=[
            ChatToolCall(
                id="call_1",
                type="function",
                function=ChatFunction(
                    name="HassTurnOn",
                    arguments={"name": "Desk light"},
                ),
            )
        ],
    ),
    ChatMessage(
        role="tool",
        content='{"success": ["light.desk"]}',
        tool_call_id="call_1",
        tool_name="HassTurnOn",
    ),
]

MOCK_MESSAGE_CONTEXT_OVERFLOW: list[ChatMessage] = [
    ChatMessage(role="system", content="You are a helpful assistant."),
    ChatMessage(
        role="user",
        content="".join(f"Message Overflow INDEX: {i}" for i in range(10000)),
    ),
]
