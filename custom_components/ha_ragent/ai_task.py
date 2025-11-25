from __future__ import annotations
import logging

from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .src.homeassistant.ragent_client import RAGentConfigEntry
from .src.homeassistant.ragent_task import RAGentTaskEntity

_logger = logging.getLogger(__name__)

async def async_setup_entry(hass: HomeAssistant, config_entry: RAGentConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    """Set up AI Task entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "ai_task_data":
            continue

        async_add_entities(
            [RAGentTaskEntity(hass, config_entry, subentry, config_entry.runtime_data)],
            config_subentry_id=subentry.subentry_id,
        )