from __future__ import annotations
import logging

from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .src.homeassistant.ragent_task import RAGentTaskEntity

_logger = logging.getLogger(__name__)

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    """Set up HA Ragent AI Task from a config entry."""
    for subentry in config_entry.subentries.values():
        async_add_entities(
            [RAGentTaskEntity(hass, config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )