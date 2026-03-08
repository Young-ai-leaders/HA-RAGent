from typing import List, Dict, Any
import json
from dataclasses import dataclass

@dataclass
class Device:
    id: str
    name: str
    area_name: str
    domain: List[str] = None
    device_tags: List[str] = None
    services: List[str] = None
    state: str = None
    attributes: Dict[str, Any] = None
    
    def __str__(self):
        return json.dumps({
            "device_id": self.id,
            "name": self.name,
            "domain": self.domain,
            "area_name": self.area_name,
            "device_tags": self.device_tags,
            "services": self.services,
        })