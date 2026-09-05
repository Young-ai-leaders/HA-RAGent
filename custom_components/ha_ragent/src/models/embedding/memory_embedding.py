from __future__ import annotations

from typing import Any

from custom_components.ha_ragent.src.models.base.embedding_record import EmbeddingRecord
from custom_components.ha_ragent.src.models.embedding.memory import Memory

class MemoryEmbedding(EmbeddingRecord[Memory]):
    def __init__(self, memory: Memory, vector_embedding: list[float]) -> None:
        super().__init__(memory, vector_embedding)

    def object_to_dict(self) -> dict[str, Any]:
        """Return the persisted fields for the embedded memory."""
        return self.embedded_object.to_dict()

    @staticmethod
    def parse_object(doc: dict[str, Any]) -> Memory:
        """Restore the memory object from a persisted vector record."""
        return Memory.from_dict(doc)
