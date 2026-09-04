from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TurnContext:
    key: str
    text: str
    entities: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    areas: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()
    device_classes: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    ambiguous_entities: tuple[str, ...] = ()
    created_at: float | None = None

    def to_embedding_text(self) -> str:
        values = (
            self.text,
            *self.entities,
            *self.tools,
            *self.areas,
            *self.domains,
            *self.device_classes,
            *self.actions,
            *self.ambiguous_entities,
        )
        return " | ".join(value for value in values if value)

    @property
    def has_canonical_context(self) -> bool:
        return bool(self.entities or self.tools or self.areas or self.domains or self.device_classes)


@dataclass
class ContinuityContext:
    entities: dict[str, float] = field(default_factory=dict)
    tools: dict[str, float] = field(default_factory=dict)
    areas: dict[str, float] = field(default_factory=dict)
    domains: dict[str, float] = field(default_factory=dict)
    device_classes: dict[str, float] = field(default_factory=dict)
    actions: dict[str, float] = field(default_factory=dict)
    ambiguous_entities: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def _maximum(values: dict[str, float], candidates: list[str]) -> float:
        return max((values.get(str(candidate).casefold(), 0.0) for candidate in candidates if candidate), default=0.0)

    def device_score(self, device: object) -> float:
        """Return a small continuity boost for a device candidate."""
        entity_id = str(getattr(device, "id", "") or "")
        area = str(getattr(device, "area_name", "") or "")
        domains = list(getattr(device, "domain", None) or [])
        device_class = str(getattr(device, "device_class", "") or "")
        return (
            1.5 * self._maximum(self.entities, [entity_id])
            + 0.75 * self._maximum(self.areas, [area])
            + 0.5 * self._maximum(self.domains, domains)
            + 0.5 * self._maximum(self.device_classes, [device_class])
            + 0.2 * self._maximum(self.ambiguous_entities, [entity_id])
        )

    def tool_score(self, tool: object) -> float:
        """Return a continuity boost for a tool candidate."""
        name = str(getattr(tool, "name", "") or "")
        return self._maximum(self.tools, [name]) + 0.5 * self._maximum(self.actions, [name])
