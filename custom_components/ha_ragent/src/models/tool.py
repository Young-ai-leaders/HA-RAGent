from typing import List, Dict, Any
import json
from dataclasses import asdict, dataclass, is_dataclass

from custom_components.ha_ragent.src.models.tool_metadata import ToolMetadata

@dataclass
class LlmTool:
    name: str
    description: str
    metadata: ToolMetadata = None
    parameters: Dict[str, Any] = None
    
    def __str__(self):
        return self.to_json()

    def to_embedding_text(self):
        return json.dumps({
            "name": self.name,
            "description": self.description
        })

    def to_json(self):
        return json.dumps({
            "name": self.name,
            "description": self.description,
            "parameters": json.dumps(self.parameters),
            "metadata": self.metadata.to_json()
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
