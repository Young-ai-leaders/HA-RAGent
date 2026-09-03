from typing import Any

from custom_components.ha_ragent.src.models.embedding_record import EmbeddingRecord
from custom_components.ha_ragent.src.models.device import Device

class DeviceEmbedding(EmbeddingRecord[Device]):
    def __init__(self, device: Device, vector_embedding: list[float]) -> None:
        self.device = device
        super().__init__(device, vector_embedding)

    def object_to_dict(self) -> dict[str, Any]:
        return self.device.to_dict()

    @staticmethod
    def parse_object(doc: dict[str, Any]) -> Device:
        return Device(
            id=doc.get("device_id"),
            friendly_name=doc.get("friendly_name"),
            domain=doc.get("domain"),
            floor_name=doc.get("floor_name"),
            area_name=doc.get("area_name"),
            device_labels=doc.get("device_labels", []),
            services=doc.get("services", []),
            aliases=doc.get("aliases", []),
            unit_of_measurement=doc.get("unit_of_measurement")
        )
