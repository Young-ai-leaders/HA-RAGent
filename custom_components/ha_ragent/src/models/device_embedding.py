from typing import List, Dict
from models.device import Device

class DeviceEmbedding:
    def __init__(self, device: Device, vector_embedding: List[float]) -> None:
        self.device = device
        self.vector_embedding = vector_embedding
    
    def to_dict(self) -> Dict:
        return {
                "device_id": self.device.id,
                "name": self.device.name,
                "type": self.device.type,
                "location": self.device.location,
                "description": self.device.description,
                "vector_embedding": self.vector_embedding
            }    