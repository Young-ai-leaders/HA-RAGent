from __future__ import annotations

from typing import Any, Dict
import json
from dataclasses import dataclass

from custom_components.ha_ragent.src.models.base.serializeable_model import SerializableModel
from custom_components.ha_ragent.src.models.base.embeddable_model import EmbeddableModel
from custom_components.ha_ragent.src.models.embedding.tool_metadata import (
    ToolMetadata,
    canonical_action_aliases as get_canonical_action_aliases,
    canonical_action_from_text,
    split_canonical_name,
)

@dataclass
class LlmTool(SerializableModel, EmbeddableModel):
    name: str
    description: str
    metadata: ToolMetadata = None
    parameters: Dict[str, Any] = None

    @property
    def canonical_name_parts(self) -> tuple[str, ...]:
        """Get the canonical parts of the tool's name."""
        return self.split_canonical_name(self.name)

    @property
    def canonical_action_keywords(self) -> tuple[str, ...]:
        """Return the action-bearing canonical name parts."""
        parts = self.canonical_name_parts
        return parts[1:] if parts and parts[0] == "hass" else parts

    @property
    def canonical_action(self) -> str:
        """Return the tool's canonical action capability."""
        searchable_text = " ".join((*self.canonical_name_parts, self.description or ""))
        return canonical_action_from_text(searchable_text)

    @property
    def canonical_action_aliases(self) -> tuple[str, ...]:
        """Return equivalent phrases for the tool's canonical action."""
        return get_canonical_action_aliases(self.canonical_action)

    @staticmethod
    def _schema_values(schema: object) -> set[str]:
        values: set[str] = set()
        if isinstance(schema, dict):
            for name, value in schema.items():
                if name == "const" and isinstance(value, (str, int, float)):
                    values.add(str(value).casefold())
                elif name == "enum" and isinstance(value, list):
                    values.update(str(item).casefold() for item in value)
                else:
                    values.update(LlmTool._schema_values(value))
        elif isinstance(schema, list):
            for value in schema:
                values.update(LlmTool._schema_values(value))
        return values

    @property
    def canonical_supported_domains(self) -> tuple[str, ...]:
        """Return explicit target domains declared by the tool schema."""
        properties = (self.parameters or {}).get("properties") or {}
        return tuple(sorted(self._schema_values(properties.get("domain", {}))))

    @property
    def canonical_search_parts(self) -> tuple[str, ...]:
        """Return phrase-oriented text used to retrieve this tool."""
        canonical_name = " ".join(self.canonical_name_parts)
        action_keywords = " ".join(self.canonical_action_keywords)
        return tuple(
            value
            for value in (
                self.name,
                canonical_name,
                action_keywords,
                *self.canonical_action_aliases,
                *self.canonical_supported_domains,
                self.family,
                self.description,
            )
            if value
        )

    @property
    def family(self) -> str:
        """Determine the family of the tool based on its metadata or name."""
        if self.metadata and self.metadata.family:
            return self.metadata.family
        return ToolMetadata.family_from_name(self.name)

    @staticmethod
    def split_canonical_name(name: str) -> tuple[str, ...]:
        """Split on underscores and camel-case transitions."""
        return split_canonical_name(name)
    
    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the tool."""
        return {
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata.to_json_str() if self.metadata else None,
            "parameters": json.dumps(self.parameters)
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'LlmTool':
        """Create an LlmTool instance from a dictionary."""
        metadata = json.loads(data["metadata"]) if data.get("metadata") else None
        parameters = json.loads(data["parameters"]) if data.get("parameters") else None
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            metadata=ToolMetadata.from_dict(metadata) if metadata else None,
            parameters=parameters,
        )

    def to_embedding_text(self) -> str:
        """Return a string representation of the tool for embedding purposes."""
        parts = [ f"Tool name: {self.name}" ]
        self.append_if_exists(
            parts,
            "canonical parts",
            " ".join(self.canonical_name_parts),
        )
        self.append_if_exists(parts, "family", self.family)
        self.append_if_exists(parts, "action keywords", " ".join(self.canonical_action_keywords))
        self.append_if_exists(parts, "action aliases", ", ".join(self.canonical_action_aliases))
        self.append_if_exists(
            parts,
            "supported domains",
            ", ".join(self.canonical_supported_domains),
        )
        self.append_if_exists(parts, "Description", self.description)

        return " | ".join(parts)

    def to_tool_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of the tool suitable for use in an LLM context."""
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
