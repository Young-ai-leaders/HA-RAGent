import logging

from homeassistant.const import Platform, EVENT_HOMEASSISTANT_STARTED
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.components.homeassistant.exposed_entities import async_should_expose

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

PLATFORMS = (Platform.CONVERSATION, Platform.AI_TASK)

def _create_vector_db_client(hass: HomeAssistant, vector_db_backend_type: str, entry: ConfigEntry) -> ABaseDbBackend:
    _logger.debug("Creating Vector DB client of type %s", vector_db_backend_type)
    return vector_db_to_class(vector_db_backend_type)(hass, dict(entry.options))

def _create_embedding_client(hass: HomeAssistant, embedding_backend_type: str, entry: ConfigEntry) -> ABaseEmbedder:
    _logger.debug("Creating Embedding client of type %s", embedding_backend_type)
    return embedding_backend_to_class(embedding_backend_type)(hass, dict(entry.options))

def _create_llm_client(hass: HomeAssistant, llm_backend_type: str, entry: ConfigEntry) -> ALlmBaseBackend:
    _logger.debug("Creating LLM client of type %s", llm_backend_type)
    return llm_backend_to_class(llm_backend_type)(hass, dict(entry.options))

async def _async_embed_all_exposed_devices(hass: HomeAssistant, entry: ConfigEntry) -> None:
    _logger.info("===== DEVICE EMBEDDING FUNCTION CALLED =====")
    try:
        _logger.debug("Device embedding function starting, checking for subentries")
        if not hasattr(entry, "subentries") or not entry.subentries:
            _logger.debug("No subentries found in config entry! Cannot embed devices.")
            return

        _logger.debug(f"Found {len(entry.subentries)} subentries to process.")

        all_entities = list(hass.states.async_entity_ids())
        exposed_entities = [entity_id for entity_id in all_entities if async_should_expose(hass, "conversation", entity_id)]
        _logger.debug(f"Device embedding starting: {len(all_entities)} total entities, {len(exposed_entities)} exposed to conversation.")

        if not exposed_entities:
            _logger.warning("No entities are exposed to Conversation. Skipping embedding and preserving existing vectors.")
            return

        for subentry_id, subentry in entry.subentries.items():
            try:
                collection_name = f"devices_{subentry_id}"
                embedding_len = len(await entry.embedder_backend.async_embed_text(dict(subentry.data), "Test"))
                
                _logger.debug(f"Preparing collection reset for subentry {subentry_id} (collection: {collection_name}, embedding_len: {embedding_len}).")
                await entry.vector_db_backend.async_reset_database(dict(subentry.data), collection_name, embedding_len)
                _logger.debug(f"Collection reset done. Starting embedding job for subentry {subentry_id} (collection: {collection_name}).")
                
                device_list = await DeviceExtractor(hass).async_get_embeddable_devices(exposed_entities)
                device_embeddings = await entry.embedder_backend.async_embed_devices(dict(subentry.data), device_list)

                if device_embeddings:
                    _logger.debug(f"Saving {len(device_embeddings)} device embeddings to collection {collection_name}.")
                    await entry.vector_db_backend.async_save_device_embeddings(
                        dict(subentry.data),
                        collection_name,
                        device_embeddings,
                    )
                    _logger.debug(f"Finished embedding all exposed devices for subentry {subentry_id} ({len(device_embeddings)} devices)")
                else:
                    _logger.warning("No devices to embed for subentry %s", subentry_id)
            except Exception as err:
                _logger.error(f"Error in background embedding job for subentry {subentry_id}: {err}", exc_info=True)
                continue
    except Exception as err:
        _logger.error("Error in background embedding job: %s", err, exc_info=True)
    finally:
        _logger.info("===== DEVICE EMBEDDING FUNCTION FINISHED =====")

async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    await hass.config_entries.async_reload(entry.entry_id)

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry):
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

    if hass.is_running:
        hass.async_create_task(_async_embed_all_exposed_devices(hass, entry))
    else:
        hass.bus.async_listen_once(
            EVENT_HOMEASSISTANT_STARTED,
            lambda _event: hass.add_job(_async_embed_all_exposed_devices(hass, entry)),
        )

    entry.async_on_unload(entry.add_update_listener(_async_update_listener))
    return True
    
async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    if not await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        return False
    
    hass.data[DOMAIN].pop(entry.entry_id)
    return True


