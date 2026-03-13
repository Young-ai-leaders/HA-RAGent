from typing import Any, List, Dict

from .tool import LlmTool

class LlmToolEmbedding:
    def __init__(self, tool: LlmTool, vector_embedding: List[float]) -> None:
        self.tool = tool
        self.vector_embedding = vector_embedding
    
    def to_dict(self) -> Dict:
        return {
                "tool_id": self.tool.id,
                "description": self.tool.description,
                "vector_embedding": self.vector_embedding
            }
    
    @staticmethod
    def from_dict(doc: Dict[str, Any]) -> LlmTool:
        return LlmTool(
            id=doc.get("tool_id"),
            description=doc.get("description")
        )