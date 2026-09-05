from dataclasses import dataclass
from typing import Any

from custom_components.ha_ragent.src.models.base.serializeable_model import SerializableModel

@dataclass
class ModelInfo(SerializableModel):
    name: str
    context_size: int | None
    is_embedding_model: bool | None
    is_tool_model: bool | None

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the model info."""
        return {
            "name": self.name,
            "context_size": self.context_size,
            "is_embedding_model": self.is_embedding_model,
            "is_tool_model": self.is_tool_model
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ModelInfo':
        """Create a ModelInfo instance from a dictionary."""
        return cls(
            name=data.get("name", ""),
            context_size=data.get("context_size"),
            is_embedding_model=data.get("is_embedding_model"),
            is_tool_model=data.get("is_tool_model")
        )
