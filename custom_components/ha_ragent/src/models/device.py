from dataclasses import dataclass, asdict
import json

@dataclass
class SmartHomeDevice:
    id: str
    name: str
    type: str
    location: str
    description: str = ""
    capabilities: list[str] = None
    
    def __str__(self):
         return json.dumps(asdict(self))