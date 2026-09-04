from abc import ABC, abstractmethod
import json

class SerializableModel(ABC):
    @abstractmethod
    def to_dict(self) -> dict[str, any]:
        """Return a dictionary representation of the embeddable model."""
        raise NotImplementedError("to_dict method must be implemented in subclasses.")

    @abstractmethod
    def from_dict(cls, data: dict[str, any]) -> "SerializableModel":
        """Create an instance of the database model from a dictionary representation."""
        raise NotImplementedError("from_dict method must be implemented in subclasses.")

    def from_json_str(cls, json_str: str) -> "SerializableModel":
        """Create an instance of the database model from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def to_json_str(self) -> str:
        """Serialize the model to a JSON string."""
        return json.dumps(self.to_dict())
