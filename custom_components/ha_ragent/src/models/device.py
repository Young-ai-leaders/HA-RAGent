from typing import List, Dict, Any
import json
from dataclasses import dataclass

@dataclass
class Device:
    id: str
    name: str
    area_name: str
    floor_name: str
    domain: List[str] = None
    device_labels: List[str] = None
    services: List[str] = None
    aliases: List[str] = None

    # Loaded from current state not used for embedding
    state: str = None
    attributes: Dict[str, Any] = None

    
    def __str__(self):
        return self.to_json()

    def to_embedding_text(self):
        return json.dumps({
            "name": self.name,
            "domain": self.domain,
            "area_name": self.area_name,
            "floor_name": self.floor_name,
            "device_labels": self.device_labels,
            "aliases": self.aliases
        })

    def to_json(self):
        return json.dumps({
            "device_id": self.id,
            "name": self.name,
            "domain": self.domain,
            "area_name": self.area_name,
            "floor_name": self.floor_name,
            "device_labels": self.device_labels,
            "services": self.services,
            "aliases": self.aliases
        })