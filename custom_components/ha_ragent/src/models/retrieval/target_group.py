from __future__ import annotations
from dataclasses import dataclass

@dataclass
class TargetGroup:
    entities: tuple[str, ...] = ()
    areas: tuple[str, ...] = ()
    floors: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()
    device_classes: tuple[str, ...] = ()
    tool: str = ""
    action: str = ""