import json
import re
from dataclasses import dataclass
from typing import Any

from custom_components.ha_ragent.src.models.base.serializeable_model import SerializableModel

CANONICAL_NAME_SPLIT_PATTERN = r"_|(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
CANONICAL_ACTION_ALIASES: dict[str, tuple[str, ...]] = {
    "off": ("turn off", "switch off", "power off", "shut off", "disable"),
    "on": ("turn on", "switch on", "power on", "enable"),
    "toggle": ("toggle",),
    "unlock": ("unlock",),
    "lock": ("lock",),
    "open": ("open",),
    "close": ("close",),
    "brightness": ("set brightness", "dim"),
    "temperature": ("set temperature",),
    "pause": ("pause",),
    "play": ("play", "resume"),
    "volume": ("set volume", "mute", "unmute"),
    "position": ("set position", "position"),
    "cancel": ("cancel",),
}


def split_canonical_name(name: str) -> tuple[str, ...]:
    """Split a canonical name on underscores and camel-case transitions."""
    return tuple(
        part.casefold()
        for part in re.split(CANONICAL_NAME_SPLIT_PATTERN, str(name or ""))
        if part
    )


def normalize_canonical_text(text: object) -> str:
    """Normalize free text for canonical capability matching."""
    value = re.sub(CANONICAL_NAME_SPLIT_PATTERN, " ", str(text or "")).casefold()
    return " ".join(
        "".join(character if character.isalnum() else " " for character in value).split()
    )


def canonical_action_from_text(text: object) -> str:
    """Return the canonical action explicitly expressed by text."""
    normalized_text = normalize_canonical_text(text)
    if normalized_text in CANONICAL_ACTION_ALIASES:
        return normalized_text
    normalized = f" {normalized_text} "
    for action, aliases in CANONICAL_ACTION_ALIASES.items():
        if any(f" {alias} " in normalized for alias in aliases):
            return action
    tokens = normalized_text.split()
    directional_verbs = {"power", "shut", "switch", "turn"}
    for action in ("off", "on"):
        for index, token in enumerate(tokens):
            if token != action:
                continue
            recent_tokens = set(tokens[max(0, index - 3):index])
            if index == len(tokens) - 1 or recent_tokens & directional_verbs:
                return action
    return ""


def canonical_action_aliases(text: object) -> tuple[str, ...]:
    """Return search aliases for the canonical action expressed by text."""
    action = canonical_action_from_text(text)
    return CANONICAL_ACTION_ALIASES.get(action, ())


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
