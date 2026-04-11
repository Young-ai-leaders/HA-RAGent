from typing import List, Dict, Any
import json
from dataclasses import dataclass

@dataclass
class Device:
    id: str
    name: str
    area_name: str
    domain: List[str] = None
    device_labels: List[str] = None
    services: List[str] = None
    aliases: List[str] = None

    # Loaded from current state not used for embedding
    state: str = None
    attributes: Dict[str, Any] = None

    
    def __str__(self):
        return self.to_json()

    def to_json(self):
        return json.dumps({
            "device_id": self.id,
            "name": self.name,
            "domain": self.domain,
            "area_name": self.area_name,
            "device_labels": self.device_labels,
            "services": self.services,
            "aliases": self.aliases
        })