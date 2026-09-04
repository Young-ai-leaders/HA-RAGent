from typing import Any

from custom_components.ha_ragent.src.models.base.embedding_record import EmbeddingRecord
from custom_components.ha_ragent.src.models.embedding.tool import LlmTool

class LlmToolEmbedding(EmbeddingRecord[LlmTool]):
    def __init__(self, tool: LlmTool, vector_embedding: list[float]) -> None:
        super().__init__(tool, vector_embedding)

    def object_to_dict(self) -> dict[str, Any]:
        """Return the persisted fields for the embedded tool."""
        return self.embedded_object.to_dict()
    
    @staticmethod
    def parse_object(doc: dict[str, Any]) -> LlmTool:
        """Restore the tool object from a persisted vector record."""
        return LlmTool.from_dict(doc)
