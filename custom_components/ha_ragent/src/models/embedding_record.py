from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from custom_components.ha_ragent.src.models.embeddable_model import EmbeddableModel
from custom_components.ha_ragent.src.models.serializable_model import SerializableModel


T = TypeVar("T", bound=EmbeddableModel)


class EmbeddingRecord(SerializableModel, Generic[T], ABC):
    def __init__(self, embedded_object: T, vector_embedding: list[float]) -> None:
        self.embedded_object = embedded_object
        self.vector_embedding = vector_embedding

    @abstractmethod
    def object_to_dict(self) -> dict[str, Any]:
        """Return the persisted fields for the embedded object."""

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.object_to_dict(),
            "vector_embedding": self.vector_embedding,
        }

    @staticmethod
    @abstractmethod
    def parse_object(doc: dict[str, Any]) -> T:
        """Restore the domain object from a persisted vector record."""
