import logging
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from custom_components.ha_ragent.src.llm_backends.base_backend import ALlmBaseBackend

from .src.homeassistant.ragent_client import RAGentConfigEntry

from .src.const import (
    BACKEND_TO_CLASS,
    DOMAIN,
    
    CONF_LLM_BACKEND_TYPE,
    
    DEFAULT_LLM_BACKEND_TYPE,    
)

_logger = logging.getLogger(__name__)

PLATFORMS = (Platform.CONVERSATION, Platform.AI_TASK)

def create_client(hass: HomeAssistant, backend_type: str, entry: RAGentConfigEntry) -> ALlmBaseBackend:
    _logger.debug("Creating LLM client of type %s", backend_type)
    return BACKEND_TO_CLASS[backend_type](hass, dict(entry.options))

async def async_setup_entry(hass: HomeAssistant, entry: RAGentConfigEntry):
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = entry
    
    backend_type = entry.data.get(CONF_LLM_BACKEND_TYPE, DEFAULT_LLM_BACKEND_TYPE)
    entry.runtime_data = await hass.async_add_executor_job(create_client, hass, backend_type, entry)

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))
    return True
    
async def _async_update_listener(hass: HomeAssistant, entry: RAGentConfigEntry) -> None:
    await hass.config_entries.async_reload(entry.entry_id)

async def async_unload_entry(hass: HomeAssistant, entry: RAGentConfigEntry) -> bool:
    if not await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        return False
    hass.data[DOMAIN].pop(entry.entry_id)
    return True