from typing import Any, List, Dict

from .device import Device

class DeviceEmbedding:
    def __init__(self, device: Device, vector_embedding: List[float]) -> None:
        self.device = device
        self.vector_embedding = vector_embedding
    
    def to_dict(self) -> Dict:
        return {
                "device_id": self.device.id,
                "name": self.device.name,
                "domain": self.device.domain,
                "area_name": self.device.area_name,
                "device_tags": self.device.device_tags,
                "services": self.device.services,
                "vector_embedding": self.vector_embedding
            }

    @staticmethod
    def parse_object(doc: Dict[str, Any]) -> Device:
        return Device(
            id=doc.get("device_id"),
            name=doc.get("name"),
            domain=doc.get("domain"),
            area_name=doc.get("area_name"),
            device_tags=doc.get("device_tags", []),
            services=doc.get("services", [])
        )