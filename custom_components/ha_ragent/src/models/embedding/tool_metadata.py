import json
import re
from dataclasses import dataclass
from typing import Any

from custom_components.ha_ragent.src.const import CANONICAL_ACTION_ALIASES, CANONICAL_NAME_SPLIT_PATTERN
from custom_components.ha_ragent.src.models.base.serializeable_model import SerializableModel


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
    actions = canonical_actions_from_text(text)
    return actions[0] if actions else ""


def canonical_actions_from_text(text: object) -> tuple[str, ...]:
    """Return explicit canonical actions in their textual order."""
    normalized_text = normalize_canonical_text(text)
    if normalized_text in CANONICAL_ACTION_ALIASES:
        return (normalized_text,)
    normalized = f" {normalized_text} "
    matches: list[tuple[int, str]] = []
    for action, aliases in CANONICAL_ACTION_ALIASES.items():
        for alias in aliases:
            marker = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)")
            matches.extend((match.start(), action) for match in marker.finditer(normalized_text))
    tokens = normalized_text.split()
    directional_verbs = {"power", "shut", "switch", "turn"}
    informational_words = {"are", "check", "is", "status", "what", "whether", "which"}
    for action in ("off", "on"):
        if any(found_action == action for _, found_action in matches):
            continue
        for index, token in enumerate(tokens):
            if token != action:
                continue
            recent_tokens = set(tokens[max(0, index - 3):index])
            is_short_imperative = index == len(tokens) - 1 and not set(tokens) & informational_words
            if is_short_imperative or recent_tokens & directional_verbs:
                matches.append((index, action))
                break
    return tuple(action for _, action in sorted(set(matches)))


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
