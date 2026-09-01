from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Memory:
    id: str
    content: str
    created_at: str

    def to_embedding_text(self) -> str:
        return json.dumps({"memory": self.content})

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "content": self.content,
            "created_at": self.created_at,
        }
