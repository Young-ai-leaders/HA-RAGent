from typing import Dict, Any
import json
from dataclasses import dataclass

from custom_components.ha_ragent.src.models.embeddable_model import EmbeddableModel
from custom_components.ha_ragent.src.models.tool_metadata import ToolMetadata

@dataclass
class LlmTool(EmbeddableModel):
    name: str
    description: str
    metadata: ToolMetadata = None
    parameters: Dict[str, Any] = None
    
    def to_embedding_text(self) -> str:
        parts = [f"Tool name: {self.name}"]
        
        if self.description:
            parts.append(f"description: {self.description}")

        if self.parameters:
            parts.append(
                "parameters: "
                + json.dumps(self.parameters, ensure_ascii=False, separators=(",", ":"))
            )

        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": json.dumps(self.parameters),
            "metadata": self.metadata.to_json() if self.metadata else None,
        }

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
