import logging
from typing import Final
from homeassistant import core
from homeassistant.const import ATTR_ENTITY_ID, Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as llm
from homeassistant.util.json import JsonObjectType
from homeassistant.config_entries import ConfigEntry

from .src.const import (
    ALLOWED_SERVICE_CALL_ARGUMENTS,
    DOMAIN,
    LLM_API_ID,
    CONF_BACKEND_TYPE,
    DEFAULT_BACKEND_TYPE,
    BACKEND_TO_CLASS,
    SERVICE_TOOL_NAME,
    SERVICE_TOOL_ALLOWED_SERVICES,
    SERVICE_TOOL_ALLOWED_DOMAINS,
)

_logger = logging.getLogger(__name__)

PLATFORMS = (Platform.CONVERSATION, Platform.AI_TASK)

async def async_setup(hass: HomeAssistant, config: dict):
    """Set up integration from YAML (if used)."""
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry):
    """Set up integration from a config entry."""
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = entry.data
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry):
    """Unload a config entry."""
    hass.data[DOMAIN].pop(entry.entry_id, None)
    return True
