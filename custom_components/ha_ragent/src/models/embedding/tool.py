from typing import Dict, Any
import json
import re
from dataclasses import dataclass

from custom_components.ha_ragent.src.models.base.database_model import DatabaseModel
from custom_components.ha_ragent.src.models.base.embeddable_model import EmbeddableModel
from custom_components.ha_ragent.src.models.embedding.tool_metadata import ToolMetadata

REGEX_SPLIT_PATTERN = r"_|(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"

@dataclass
class LlmTool(DatabaseModel, EmbeddableModel):
    name: str
    description: str
    metadata: ToolMetadata = None
    parameters: Dict[str, Any] = None

    @property
    def name_parts(self) -> tuple[str, ...]:
        """Get the canonical parts of the tool's name."""
        return self.split_canonical_name(self.name)

    @property
    def family(self) -> str:
        """Determine the family of the tool based on its metadata or name."""
        if self.metadata and self.metadata.family:
            return self.metadata.family
        return ToolMetadata.family_from_name(self.name)

    @staticmethod
    def split_canonical_name(name: str) -> tuple[str, ...]:
        """Split on underscores and camel-case transitions."""
        return tuple(part.casefold() for part in re.split(REGEX_SPLIT_PATTERN, str(name or "")) if part)
    
    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the tool."""
        return {
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata.to_json() if self.metadata else None,
            "parameters": json.dumps(self.parameters)
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'LlmTool':
        """Create an LlmTool instance from a dictionary."""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            metadata=ToolMetadata.from_dict(json.loads(data.get("metadata"))) if data.get("metadata") else None,
            parameters=json.loads(data.get("parameters")) if data.get("parameters") else None
        )

    def to_embedding_text(self) -> str:
        """Return a string representation of the tool for embedding purposes."""
        parts = [ f"Tool name: {self.name}" ]
        self.append_if_exists(parts, "Canonical parts", self.name_parts)
        self.append_if_exists(parts, "Family", self.family)
        self.append_if_exists(parts, "Description", self.description)

        return " | ".join(parts)

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
