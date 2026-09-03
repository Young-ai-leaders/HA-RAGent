from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from custom_components.ha_ragent.src.models.embeddable_model import EmbeddableModel


@dataclass
class Memory(EmbeddableModel):
    id: str
    content: str
    created_at: str
    retrieval_count: int = 0

    def to_embedding_text(self) -> str:
        return f"Content: {self.content}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "created_at": self.created_at,
            "retrieval_count": self.retrieval_count,
        }
