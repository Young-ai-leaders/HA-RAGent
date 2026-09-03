from __future__ import annotations

from typing import Any

from custom_components.ha_ragent.src.models.embedding_record import EmbeddingRecord
from custom_components.ha_ragent.src.models.memory import Memory


class MemoryEmbedding(EmbeddingRecord[Memory]):
    def __init__(self, memory: Memory, vector_embedding: list[float]) -> None:
        self.memory = memory
        super().__init__(memory, vector_embedding)

    def object_to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory.id,
            "content": self.memory.content,
            "created_at": self.memory.created_at,
            "retrieval_count": self.memory.retrieval_count,
        }

    @staticmethod
    def parse_object(doc: dict[str, Any]) -> Memory:
        return Memory(
            id=str(doc.get("memory_id", "")),
            content=str(doc.get("content", "")),
            created_at=str(doc.get("created_at", "")),
            retrieval_count=int(doc.get("retrieval_count", 0) or 0),
        )
