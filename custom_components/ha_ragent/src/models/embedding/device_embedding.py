from typing import Any

from custom_components.ha_ragent.src.models.base.embedding_record import EmbeddingRecord
from custom_components.ha_ragent.src.models.embedding.device import Device

class DeviceEmbedding(EmbeddingRecord[Device]):
    def __init__(self, device: Device, vector_embedding: list[float]) -> None:
        super().__init__(device, vector_embedding)

    def object_to_dict(self) -> dict[str, Any]:
        return self.embedded_object.to_dict()

    @staticmethod
    def parse_object(doc: dict[str, Any]) -> Device:
        return Device.from_dict(doc)
