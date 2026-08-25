import json
from dataclasses import dataclass

@dataclass
class ToolMetadata:
    is_domain_aware: bool = False
    is_area_aware: bool = False
    is_device_class_aware: bool = False

    def __str__(self):
        return self.to_json()

    def to_json(self):
        return json.dumps({
            "is_domain_aware": self.is_domain_aware,
            "is_area_aware": self.is_area_aware,
            "is_device_class_aware": self.is_device_class_aware
        })

    @staticmethod
    def from_json(json_str: str) -> 'ToolMetadata':
        data = json.loads(json_str)
        return ToolMetadata(
            is_domain_aware=data.get("is_domain_aware", False),
            is_area_aware=data.get("is_area_aware", False),
            is_device_class_aware=data.get("is_device_class_aware", False)
        )