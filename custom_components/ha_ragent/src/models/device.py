from typing import List
import json
from dataclasses import dataclass, asdict

@dataclass
class SmartHomeDevice:
    id: str
    name: str
    type: str
    location: str
    description: str = ""
    capabilities: List[str] = None
    
    def __str__(self):
         return json.dumps(asdict(self))