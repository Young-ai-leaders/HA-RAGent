from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any


class SerializableModel(ABC):
    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Return the stable representation used for serialization."""

    def to_json(self) -> str:
        """Serialize the model without ASCII-escaping human-readable text."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def __str__(self) -> str:
        return self.to_json()
