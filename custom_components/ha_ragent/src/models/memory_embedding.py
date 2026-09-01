from __future__ import annotations

from typing import Any

from custom_components.ha_ragent.src.models.memory import Memory


class MemoryEmbedding:
    def __init__(self, memory: Memory, vector_embedding: list[float]) -> None:
        self.memory = memory
        self.vector_embedding = vector_embedding

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory.id,
            "content": self.memory.content,
            "created_at": self.memory.created_at,
            "vector_embedding": self.vector_embedding,
        }

    @staticmethod
    def parse_object(doc: dict[str, Any]) -> Memory:
        return Memory(
            id=str(doc.get("memory_id", "")),
            content=str(doc.get("content", "")),
            created_at=str(doc.get("created_at", "")),
        )
