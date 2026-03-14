import json
from typing import Any, List, Dict

from .tool import LlmTool

class LlmToolEmbedding:
    def __init__(self, tool: LlmTool, vector_embedding: List[float]) -> None:
        self.tool = tool
        self.vector_embedding = vector_embedding
    
    def to_dict(self) -> Dict:
        return {
                "name": self.tool.name,
                "description": self.tool.description,
                "parameters": json.dumps(self.tool.parameters),
                "metadata": json.dumps(self.tool.metadata),
                "vector_embedding": self.vector_embedding
            }
    
    @staticmethod
    def parse_object(doc: Dict[str, Any]) -> 'LlmTool':
        return LlmTool(
            name=doc.get("name"),
            description=doc.get("description"),
            parameters=json.loads(doc.get("parameters")) if doc.get("parameters") else None,
            metadata=json.loads(doc.get("metadata")) if doc.get("metadata") else None
        )