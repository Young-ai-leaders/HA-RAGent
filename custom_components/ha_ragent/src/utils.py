import socket
import logging
from typing import List, Any
from .backends.database.base_backend import ABaseDbBackend
from .backends.database.mongodb_backend import MongoDbBackend
from .backends.embedder.base_backend import ABaseEmbedder
from .backends.embedder.ollama_backend import OllamaEmbedder
from .backends.llm.base_backend import ALlmBaseBackend
from .backends.llm.ollama_backend import OllamaBackend
from .const import BACKEND_VECTOR_DB_TYPE_MONGODB, BACKEND_EMBEDDING_TYPE_OLLAMA, BACKEND_LLM_TYPE_OLLAMA

_logger = logging.getLogger(__name__)

def remove_thinking_block(text: str):
    pass

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
        BACKEND_VECTOR_DB_TYPE_MONGODB: MongoDbBackend
    }
    return backend_to_class.get(vector_db_type)

def embedding_backend_to_class(backend_type: str) -> ABaseEmbedder:
    backend_to_class = {
        BACKEND_EMBEDDING_TYPE_OLLAMA: OllamaEmbedder
    }
    return backend_to_class.get(backend_type)

def llm_backend_to_class(backend_type: str) -> ALlmBaseBackend:
    backend_to_class = {
        BACKEND_LLM_TYPE_OLLAMA: OllamaBackend
    }
    return backend_to_class.get(backend_type)

def get_placeholder_translation(translations: List[str], selected_language: str) -> str:
    return translations.get(selected_language, translations["en"])