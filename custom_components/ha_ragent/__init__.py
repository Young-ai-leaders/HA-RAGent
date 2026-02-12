import logging

from homeassistant.const import Platform
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.helpers import area_registry, device_registry, entity_registry, label_registry, llm

from custom_components.ha_ragent.src.backends.database.base_backend import ABaseDbBackend
from custom_components.ha_ragent.src.backends.embedder.base_backend import ABaseEmbedder
from custom_components.ha_ragent.src.backends.llm.base_backend import ALlmBaseBackend

from .src.homeassistant.ragent import RAGent

from .src.const import (
    DOMAIN,
    
    CONF_VECTOR_DB_BACKEND_TYPE,
    CONF_EMBEDDING_BACKEND_TYPE,
    CONF_LLM_BACKEND_TYPE,
    
    DEFAULT_VECTOR_DB_BACKEND_TYPE,
    DEFAULT_EMBEDDING_BACKEND_TYPE,
    DEFAULT_LLM_BACKEND_TYPE,    
)

from .src.models.device_embedding import DeviceEmbedding
from .src.models.device import Device
from .src.utils import vector_db_to_class, embedding_backend_to_class, llm_backend_to_class

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
    try:
        area_reg = area_registry.async_get(hass)
        device_reg = device_registry.async_get(hass)
        entity_reg = entity_registry.async_get(hass)
        label_reg = label_registry.async_get(hass)

        for subentry_id, subentry in entry.subentries.items():
            try:
                collection_name = f"devices_{subentry_id}"

                await entry.vector_db_backend.async_reset_database(dict(subentry.data), collection_name, 768)
                _logger.info("Starting background embedding job for subentry %s (collection: %s)", subentry_id, collection_name)
                device_embeddings = []

                for entity_id in hass.states.async_entity_ids():
                    try:
                        # only embed exposed entities
                        if not async_should_expose(hass, "conversation", entity_id):
                            continue
                        
                        state = hass.states.get(entity_id)
                        if not state:
                            continue

                        friendly_name = state.attributes.get("friendly_name", entity_id)
                        device_type = entity_id.split(".")[0] if "." in entity_id else "unknown"

                        area_name = ""
                        entity_entry = entity_reg.async_get(entity_id)
                        if entity_entry:
                            if entity_entry.area_id:
                                area = area_reg.async_get_area(entity_entry.area_id)
                                area_name = area.name if area else ""
                            elif entity_entry.device_id:
                                device = device_reg.async_get(entity_entry.device_id)
                                if device and device.area_id:
                                    area = area_reg.async_get_area(device.area_id)
                                    area_name = area.name if area else ""

                        device_tags = []
                        if entity_entry and entity_entry.labels:
                            for label_id in entity_entry.labels:
                                label = label_reg.async_get_label(label_id)
                                if label:
                                    device_tags.append(label.name)

                        device = Device(
                            id=entity_id,
                            name=friendly_name,
                            device_type=device_type,
                            area_name=area_name,
                            device_tags=device_tags,
                            capabilities=[]
                        )

                        _logger.debug("Embedding device: %s for subentry %s", entity_id, subentry_id)
                        try:
                            embedding = await entry.embedder_backend.async_embed_text(dict(subentry.data), str(device))
                        except Exception as embed_err:
                            _logger.error("Error in async_embed_text for device %s: %s", entity_id, embed_err, exc_info=True)
                            raise

                        device_embeddings.append(
                            DeviceEmbedding(
                                device=device,
                                vector_embedding=embedding
                            )
                        )
                    except Exception as err:
                        _logger.warning("Error embedding device %s for subentry %s: %s", entity_id, subentry_id, err)
                        continue
                
                if device_embeddings:
                    _logger.info("Saving %d device embeddings to collection %s", len(device_embeddings), collection_name)
                    try:
                        await entry.vector_db_backend.async_save_device_embeddings(dict(subentry.data), collection_name, device_embeddings)
                        _logger.info("Finished embedding all exposed devices for subentry %s (%d devices)", subentry_id, len(device_embeddings))
                    except Exception as save_err:
                        _logger.error("Error saving embeddings for subentry %s: %s", subentry_id, save_err, exc_info=True)
                        raise
                else:
                    _logger.info("No devices to embed for subentry %s", subentry_id)
                    
            except Exception as err:
                _logger.error("Error in background embedding job for subentry %s: %s", subentry_id, err, exc_info=True)
                continue
            
    except Exception as err:
        _logger.error("Error in background embedding job: %s", err)

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

    hass.async_create_task(_async_embed_all_exposed_devices(hass, entry))

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))
    return True
    
async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    if not await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        return False
    
    hass.data[DOMAIN].pop(entry.entry_id)
    return True


