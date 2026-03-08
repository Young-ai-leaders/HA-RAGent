from homeassistant.config_entries import ConfigEntry

from ..backends.database.base_backend import ABaseDbBackend
from ..backends.embedder.base_backend import ABaseEmbedder
from ..backends.llm.base_backend import ALlmBaseBackend

class RAGentConfigEntry(ConfigEntry):
    """RAGent Config Entry"""
    vector_db_backend: ABaseDbBackend
    embedder_backend: ABaseEmbedder
    llm_backend: ALlmBaseBackend
