from typing import List, Dict, Any
import json
from dataclasses import dataclass

@dataclass
class LlmTool:
    name: str
    description: str
    metadata: Dict[str, Any] = None
    parameters: Dict[str, Any] = None
    
    def __str__(self):
        return self.to_json()

    def to_json(self):
        return json.dumps({
            "name": self.name,
            "description": self.description,
            "parameters": json.dumps(self.parameters),
            "metadata": json.dumps(self.metadata)
        })

    def to_tool_dict(self) -> Dict[str, Any]:
        tool_def = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description or "",
            }
        }

        if self.parameters:
            tool_def["function"]["parameters"] = self.parameters

        return tool_def