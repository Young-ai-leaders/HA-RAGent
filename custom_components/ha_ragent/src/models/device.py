from typing import List
import json
from dataclasses import dataclass, asdict

@dataclass
class Device:
    id: str
    name: str
    device_type: str
    area_name: str
    device_tags: List[str] = None
    capabilities: List[str] = None
    
    def __str__(self):
         return json.dumps(asdict(self))