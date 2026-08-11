from dataclasses import dataclass
import json

@dataclass
class ModelInfo:
    name: str
    context_size: int | None
    is_embedding_model: bool | None
    is_tool_model: bool | None

    def __str__(self):
        return self.to_json()

    def to_json(self):
        return json.dumps({
            "name": self.name,
            "context_size": self.context_size,
            "is_embedding_model": self.is_embedding_model,
            "is_tool_model": self.is_tool_model
        })