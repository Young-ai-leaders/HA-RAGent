from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from custom_components.ha_ragent.src.models.base.serializeable_model import SerializableModel
from custom_components.ha_ragent.src.models.base.embeddable_model import EmbeddableModel

T = TypeVar("T", bound=EmbeddableModel)

class EmbeddingRecord(SerializableModel, Generic[T], ABC):
    def __init__(self, embedded_object: T, vector_embedding: list[float]) -> None:
        self.embedded_object = embedded_object
        self.vector_embedding = vector_embedding

    @abstractmethod
    def object_to_dict(self) -> dict[str, Any]:
        """Return the persisted fields for the embedded object."""
        raise NotImplementedError("Subclasses must implement object_to_dict()")

    @staticmethod
    @abstractmethod
    def parse_object(doc: dict[str, Any]) -> T:
        """Restore the domain object from a persisted vector record."""
        raise NotImplementedError("Subclasses must implement parse_object()")

    def to_dict(self) -> dict[str, Any]:
        """Return the persisted fields for the embedding record."""
        return { **self.object_to_dict(), "vector_embedding": self.vector_embedding }

    @classmethod
    def from_dict(cls, doc: dict[str, Any]) -> "EmbeddingRecord[T]":
        """Restore the embedding record from a persisted document."""
        instance = cls.__new__(cls)
        instance.embedded_object = instance.parse_object(doc)
        instance.vector_embedding = doc.get("vector_embedding", [])
        return instance

    
