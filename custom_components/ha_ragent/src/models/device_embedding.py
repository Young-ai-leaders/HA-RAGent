from typing import List, Dict

from .device import Device

class DeviceEmbedding:
    def __init__(self, device: Device, vector_embedding: List[float]) -> None:
        self.device = device
        self.vector_embedding = vector_embedding
    
    def to_dict(self) -> Dict:
        return {
                "device_id": self.device.id,
                "name": self.device.name,
                "device_type": self.device.device_type,
                "area_name": self.device.area_name,
                "device_tags": self.device.device_tags,
                "vector_embedding": self.vector_embedding
            }    