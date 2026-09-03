from typing import List, Dict, Any
from dataclasses import dataclass

from custom_components.ha_ragent.src.models.embeddable_model import EmbeddableModel

@dataclass
class Device(EmbeddableModel):
    id: str
    friendly_name: str
    area_name: str
    floor_name: str
    domain: List[str] = None
    device_labels: List[str] = None
    services: List[str] = None
    aliases: List[str] = None
    unit_of_measurement: str = None

    # Loaded from current state not used for embedding
    state: str = None
    attributes: Dict[str, Any] = None
    
    def to_embedding_text(self) -> str:
        fields = {
            "entity_id": self.id,
            "friendly_name": self.friendly_name,
            "aliases": self.aliases,
            "domain": self.domain,
            "area": self.area_name,
            "floor": self.floor_name,
            "labels": self.device_labels,
            "capabilities": self.services,
            "unit": self.unit_of_measurement,
        }

        return" | ".join(
            f"{key}: {', '.join(value) if isinstance(value, list) else value}"
            for key, value in fields.items()
            if value
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.id,
            "friendly_name": self.friendly_name,
            "domain": self.domain,
            "area_name": self.area_name,
            "floor_name": self.floor_name,
            "device_labels": self.device_labels,
            "services": self.services,
            "aliases": self.aliases,
            "unit_of_measurement": self.unit_of_measurement
        }
