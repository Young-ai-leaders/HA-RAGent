from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Memory:
    id: str
    content: str
    created_at: str
    retrieval_count: int = 0

    def to_embedding_text(self) -> str:
        return json.dumps({"memory": self.content})

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "created_at": self.created_at,
            "retrieval_count": self.retrieval_count,
        }
