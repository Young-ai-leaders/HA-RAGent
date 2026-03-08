import logging
from ..models.device import Device

from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry, device_registry, entity_registry, label_registry, llm

from ..const import (
    DOMAIN
)

_logger = logging.getLogger(__name__)

class DeviceExtractor:
    def __init__(self, hass: HomeAssistant):
        self.hass = hass

    async def _async_get_services_for_domain(self, target_domain: str):
        services = self.hass.services.async_services()

        if target_domain not in services:
            return []

        return [service_name for service_name in services[target_domain]]

    async def async_get_embeddable_devices(self, exposed_entities: list[str]) -> list[Device]:
        area_reg = area_registry.async_get(self.hass)
        device_reg = device_registry.async_get(self.hass)
        entity_reg = entity_registry.async_get(self.hass)
        label_reg = label_registry.async_get(self.hass)
        
        devices = []
        
        for entity_id in exposed_entities:
            state = self.hass.states.get(entity_id)
            if not state:
                continue

            friendly_name = state.attributes.get("friendly_name", entity_id)
            domain = entity_id.split(".")[0] if "." in entity_id else "unknown"

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

            services = await self._async_get_services_for_domain(domain)

            devices.append(Device(
                id=entity_id,
                name=friendly_name,
                domain=[domain],
                area_name=area_name,
                device_tags=device_tags,
                services=services
            ))
        
        return devices

