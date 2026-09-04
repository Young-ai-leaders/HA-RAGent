import json
import re
from dataclasses import dataclass
from typing import Any

from custom_components.ha_ragent.src.models.base.serializeable_model import SerializableModel

CANONICAL_NAME_SPLIT_PATTERN = r"_|(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"


def split_canonical_name(name: str) -> tuple[str, ...]:
    """Split a canonical name on underscores and camel-case transitions."""
    return tuple(
        part.casefold()
        for part in re.split(CANONICAL_NAME_SPLIT_PATTERN, str(name or ""))
        if part
    )


@dataclass
class ToolMetadata(SerializableModel):
    family: str = None
    is_domain_aware: bool = False
    is_area_aware: bool = False
    is_device_class_aware: bool = False

    @staticmethod
    def family_from_name(name: str) -> str:
        """Map canonical Home Assistant tool-name parts to an action family."""
        if not name:
            return ""
        
        parts = set(split_canonical_name(name))
        families = (
            ("power", {"on", "off", "toggle"}),
            ("position", {"open", "close", "position"}),
            ("lock", {"lock", "unlock"}),
            ("light", {"brightness", "color", "light"}),
            ("climate", {"temperature", "climate", "thermostat"}),
            ("media", {"media", "play", "pause", "volume"}),
            ("search", {"search", "find"}),
            ("timer", {"timer"}),
        )
        return next((family for family, markers in families if parts & markers), "")

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the tool metadata."""
        return {
            "family": self.family,
            "is_domain_aware": self.is_domain_aware,
            "is_area_aware": self.is_area_aware,
            "is_device_class_aware": self.is_device_class_aware,  
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> 'ToolMetadata':
        """Create a ToolMetadata instance from a dictionary."""
        return ToolMetadata(
            family=data.get("family", None),
            is_domain_aware=data.get("is_domain_aware", False),
            is_area_aware=data.get("is_area_aware", False),
            is_device_class_aware=data.get("is_device_class_aware", False)
        )
