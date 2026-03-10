import logging

from homeassistant.const import Platform, EVENT_HOMEASSISTANT_STARTED
from homeassistant.core import HomeAssistant


from custom_components.ha_ragent.src.homeassistant.ragent_config_entry import RAGentConfigEntry
from custom_components.ha_ragent.src.backends.database.base_backend import ABaseDbBackend
from custom_components.ha_ragent.src.backends.embedder.base_backend import ABaseEmbedder
from custom_components.ha_ragent.src.backends.llm.base_backend import ALlmBaseBackend
from custom_components.ha_ragent.src.homeassistant.device_extractor import DeviceExtractor

from custom_components.ha_ragent.src.const import (
    DOMAIN,
    
    CONF_VECTOR_DB_BACKEND_TYPE,
    CONF_EMBEDDING_BACKEND_TYPE,
    CONF_LLM_BACKEND_TYPE,
    
    DEFAULT_VECTOR_DB_BACKEND_TYPE,
    DEFAULT_EMBEDDING_BACKEND_TYPE,
    DEFAULT_LLM_BACKEND_TYPE,    
)

from custom_components.ha_ragent.src.utils import vector_db_to_class, embedding_backend_to_class, llm_backend_to_class

_logger = logging.getLogger(__name__)

PLATFORMS = (Platform.CONVERSATION,)

def _create_vector_db_client(hass: HomeAssistant, vector_db_backend_type: str, entry: RAGentConfigEntry) -> ABaseDbBackend:
    _logger.debug("Creating Vector DB client of type %s", vector_db_backend_type)
    return vector_db_to_class(vector_db_backend_type)(hass, dict(entry.options))

def _create_embedding_client(hass: HomeAssistant, embedding_backend_type: str, entry: RAGentConfigEntry) -> ABaseEmbedder:
    _logger.debug("Creating Embedding client of type %s", embedding_backend_type)
    return embedding_backend_to_class(embedding_backend_type)(hass, dict(entry.options))

def _create_llm_client(hass: HomeAssistant, llm_backend_type: str, entry: RAGentConfigEntry) -> ALlmBaseBackend:
    _logger.debug("Creating LLM client of type %s", llm_backend_type)
    return llm_backend_to_class(llm_backend_type)(hass, dict(entry.options))

async def _async_update_listener(hass: HomeAssistant, entry: RAGentConfigEntry) -> None:
    await hass.config_entries.async_reload(entry.entry_id)

async def async_setup_entry(hass: HomeAssistant, entry: RAGentConfigEntry):
    """Set up HA Ragent from a config entry."""
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = entry
    
    vector_db_backend_type = entry.data.get(CONF_VECTOR_DB_BACKEND_TYPE, DEFAULT_VECTOR_DB_BACKEND_TYPE)
    embedding_backend_type = entry.data.get(CONF_EMBEDDING_BACKEND_TYPE, DEFAULT_EMBEDDING_BACKEND_TYPE)
    llm_backend_type = entry.data.get(CONF_LLM_BACKEND_TYPE, DEFAULT_LLM_BACKEND_TYPE)

    entry.vector_db_backend = _create_vector_db_client(hass, vector_db_backend_type, entry)
    entry.embedder_backend = _create_embedding_client(hass, embedding_backend_type, entry)    
    entry.llm_backend = _create_llm_client(hass, llm_backend_type, entry)

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    device_extractor = DeviceExtractor(hass, entry)
    if hass.is_running:
        hass.async_create_task(device_extractor.async_embed_all_exposed_devices())
    else:
        hass.bus.async_listen_once(
            EVENT_HOMEASSISTANT_STARTED,
            lambda _event: hass.add_job(device_extractor.async_embed_all_exposed_devices()),
        )

    entry.async_on_unload(entry.add_update_listener(_async_update_listener))
    return True
    
async def async_unload_entry(hass: HomeAssistant, entry: RAGentConfigEntry) -> bool:
    if not await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        return False

    await entry.vector_db_backend.async_cleanup_database()
    hass.data[DOMAIN].pop(entry.entry_id)
    return True


