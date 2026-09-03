import json
from typing import Any

from custom_components.ha_ragent.src.models.embedding_record import EmbeddingRecord
from custom_components.ha_ragent.src.models.tool import LlmTool
from custom_components.ha_ragent.src.models.tool_metadata import ToolMetadata

class LlmToolEmbedding(EmbeddingRecord[LlmTool]):
    def __init__(self, tool: LlmTool, vector_embedding: list[float]) -> None:
        self.tool = tool
        super().__init__(tool, vector_embedding)

    def object_to_dict(self) -> dict[str, Any]:
        return self.tool.to_dict()
    
    @staticmethod
    def parse_object(doc: dict[str, Any]) -> LlmTool:
        return LlmTool(
            name=doc.get("name"),
            description=doc.get("description"),
            parameters=json.loads(doc.get("parameters")) if doc.get("parameters") else None,
            metadata=ToolMetadata.from_json(doc.get("metadata")) if doc.get("metadata") else None
        )
