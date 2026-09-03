from dataclasses import dataclass
from typing import Any

from custom_components.ha_ragent.src.models.serializable_model import SerializableModel

@dataclass
class ModelInfo(SerializableModel):
    name: str
    context_size: int | None
    is_embedding_model: bool | None
    is_tool_model: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "context_size": self.context_size,
            "is_embedding_model": self.is_embedding_model,
            "is_tool_model": self.is_tool_model,
        }
