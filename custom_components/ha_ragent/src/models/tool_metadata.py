import json
from dataclasses import dataclass
from typing import Any

from custom_components.ha_ragent.src.models.serializable_model import SerializableModel

@dataclass
class ToolMetadata(SerializableModel):
    is_domain_aware: bool = False
    is_area_aware: bool = False
    is_device_class_aware: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_domain_aware": self.is_domain_aware,
            "is_area_aware": self.is_area_aware,
            "is_device_class_aware": self.is_device_class_aware,
        }

    @staticmethod
    def from_json(json_str: str) -> 'ToolMetadata':
        data = json.loads(json_str)
        return ToolMetadata(
            is_domain_aware=data.get("is_domain_aware", False),
            is_area_aware=data.get("is_area_aware", False),
            is_device_class_aware=data.get("is_device_class_aware", False)
        )
