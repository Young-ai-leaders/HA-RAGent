from abc import ABC, abstractmethod

from custom_components.ha_ragent.src.models.serializable_model import SerializableModel


class EmbeddableModel(SerializableModel, ABC):
    @abstractmethod
    def to_embedding_text(self) -> str:
        """Return the semantic text sent to the embedding model."""
