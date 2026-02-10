from __future__ import annotations
import logging

from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .src.homeassistant.ragent_client import RAGentConfigEntry
from .src.homeassistant.ragent import RAGent

_logger = logging.getLogger(__name__)

async def async_setup_entry(hass: HomeAssistant, entry: RAGentConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> bool:
    """Set up Local LLM Conversation from a config entry."""
    for subentry in entry.subentries.values():
        # create one agent entity per conversation subentry
        agent_entity = RAGent(hass, entry, subentry, entry.runtime_data)

        # make sure model is loaded
        await entry.runtime_data._async_load_model(dict(subentry.data))

        # register the agent entity
        async_add_entities([agent_entity], config_subentry_id=subentry.subentry_id,)

    return True