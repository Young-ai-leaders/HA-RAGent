from typing import List, Dict, Any
import json
from dataclasses import dataclass

@dataclass
class LlmTool:
    id: str
    description: str
    
    def __str__(self):
        return json.dumps({
            "device_id": self.id,
            "description": self.description,
        })