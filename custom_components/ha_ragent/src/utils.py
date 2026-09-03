import socket
import logging
import json
from importlib.resources import files
from typing import List, Any

from custom_components.ha_ragent.src.backends.database.faiss_backend import FaissDbBackend
from custom_components.ha_ragent.src.backends.database.base_backend import ABaseDbBackend
from custom_components.ha_ragent.src.backends.database.mongodb_backend import MongoDbBackend
from custom_components.ha_ragent.src.backends.database.chromadb_backend import ChromaDbBackend
from custom_components.ha_ragent.src.backends.embedder.base_backend import ABaseEmbedder
from custom_components.ha_ragent.src.backends.embedder.ollama_backend import OllamaEmbedder
from custom_components.ha_ragent.src.backends.embedder.openai_backend import OpenAiEmbedder
from custom_components.ha_ragent.src.backends.llm.base_backend import ALlmBaseBackend
from custom_components.ha_ragent.src.backends.llm.ollama_backend import OllamaLlmBackend
from custom_components.ha_ragent.src.backends.llm.openai_backend import OpenAiLlmBackend
from custom_components.ha_ragent.src.const import (
    BACKEND_VECTOR_DB_TYPE_FAISS,
    BACKEND_VECTOR_DB_TYPE_MONGODB, 
    BACKEND_VECTOR_DB_TYPE_CHROMA,
    BACKEND_EMBEDDING_TYPE_OLLAMA, 
    BACKEND_EMBEDDING_TYPE_OPENAI_COMPATIBLE,
    BACKEND_LLM_TYPE_OLLAMA, 
    BACKEND_LLM_TYPE_OPENAI_COMPATIBLE,
    DEVICE_ATTRIBUTES_TO_EXCLUDE, 
    DEVICE_ATTRIBUTES_MAX_JSON_LENGTH
)

_logger = logging.getLogger(__name__)

def get_value(value: object, default: object) -> object:
    """Returns the value when not null, otherwise the default parameter."""
    return value if value else default

def is_valid_host(host: str) -> bool:
    """Checks if the provided hostname is valid."""
    try:
        socket.gethostbyname(host)
        return True
    except socket.gaierror:
        return False

def try_parse_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except ValueError:
        return default

def vector_db_to_class(vector_db_type: str) -> ABaseDbBackend:
    backend_to_class = {
        BACKEND_VECTOR_DB_TYPE_MONGODB: MongoDbBackend,
        BACKEND_VECTOR_DB_TYPE_CHROMA: ChromaDbBackend,
        BACKEND_VECTOR_DB_TYPE_FAISS: FaissDbBackend
    }
    return backend_to_class.get(vector_db_type)

def embedding_backend_to_class(backend_type: str) -> ABaseEmbedder:
    backend_to_class = {
        BACKEND_EMBEDDING_TYPE_OLLAMA: OllamaEmbedder,
        BACKEND_EMBEDDING_TYPE_OPENAI_COMPATIBLE: OpenAiEmbedder,
    }
    return backend_to_class.get(backend_type)

def llm_backend_to_class(backend_type: str) -> ALlmBaseBackend:
    backend_to_class = {
        BACKEND_LLM_TYPE_OLLAMA: OllamaLlmBackend,
        BACKEND_LLM_TYPE_OPENAI_COMPATIBLE: OpenAiLlmBackend,
    }
    return backend_to_class.get(backend_type)

def get_placeholder_translation(translations: List[str], selected_language: str) -> str:
    return translations.get(selected_language, translations["en"])

def _read_tool_descriptions(language: str) -> dict[str, str]:
    try:
        translation_file = files("custom_components.ha_ragent").joinpath(
            "translations", f"tool_{language}.json"
        )
        translation = json.loads(translation_file.read_text(encoding="utf-8"))
        descriptions = translation
        return descriptions if isinstance(descriptions, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, ModuleNotFoundError):
        return {}

_TOOL_DESCRIPTIONS: dict[str, dict[str, str]] = {}

async def async_load_tool_descriptions(hass: Any) -> None:
    """Load tool descriptions off the event loop before tools are constructed."""
    descriptions = await hass.async_add_executor_job(
        lambda: {
            language: _read_tool_descriptions(language)
            for language in ("en", "de")
        }
    )
    _TOOL_DESCRIPTIONS.update(descriptions)

def get_tool_description(language: str | None, tool_name: str) -> str:
    selected_language = language or "en"
    descriptions = _TOOL_DESCRIPTIONS.get(selected_language, {})
    return str(descriptions.get(tool_name) or _TOOL_DESCRIPTIONS.get("en", {}).get(tool_name) or "")

def clean_device_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    cleaned_attributes = attributes.copy()
    for key, value in attributes.items():
        if key in DEVICE_ATTRIBUTES_TO_EXCLUDE:
            cleaned_attributes.pop(key)

        try:
            json_value = json.dumps(value)
            if len(json_value) > DEVICE_ATTRIBUTES_MAX_JSON_LENGTH:
                cleaned_attributes.pop(key)
        except (TypeError, OverflowError):
            cleaned_attributes.pop(key)

    return cleaned_attributes
