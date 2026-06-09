"""Tiny ROS2-shaped, in-process message bus + typed messages.

Single process, synchronous: the bus is a blackboard holding the latest message per
topic, plus optional subscriber callbacks. Mirrors ROS2 topics (/proprio, /scene, /plan,
/goal, /feedback) so this can migrate to real rclpy later without reshaping the code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np


# --- message types (the "topics") -------------------------------------------
@dataclass
class CameraFrame:
    rgb: np.ndarray            # HxWx3 uint8 (RGB)
    camera: str = "chase"


@dataclass
class SceneView:
    """What perception/Gemini-ER understood about the scene."""
    caption: str = ""                       # "what it sees"
    targets: dict[str, tuple] = field(default_factory=dict)  # name -> (x,y) world


@dataclass
class Goal:
    kind: str = "idle"         # "go_to" | "go_to_visual" | "look" | "look_at" | "skill" | "idle"
    target_xy: tuple | None = None
    skill: str | None = None
    target_name: str | None = None  # named object to recall from spatial memory (go_to/look_at)
    text: str = ""             # original user command


@dataclass
class Plan:
    steps: list[str] = field(default_factory=list)  # human-readable plan
    reasoning: str = ""                              # "what it's thinking"
    current: str = ""
    brain: str = ""            # which brain decided: "ER" | "3.5" | "local" (M3.6); "" = unset


@dataclass
class Feedback:
    pos: tuple = (0.0, 0.0, 0.0)
    yaw: float = 0.0
    height: float = 0.0
    upright: bool = True
    status: str = ""           # success-check outcome, natural language


class Bus:
    """Latest-value blackboard with optional callbacks. Not thread-safe (single loop)."""

    def __init__(self) -> None:
        self._latest: dict[str, Any] = {}
        self._subs: dict[str, list[Callable[[Any], None]]] = {}

    def publish(self, topic: str, msg: Any) -> None:
        self._latest[topic] = msg
        for cb in self._subs.get(topic, []):
            cb(msg)

    def latest(self, topic: str, default: Any = None) -> Any:
        return self._latest.get(topic, default)

    def subscribe(self, topic: str, cb: Callable[[Any], None]) -> None:
        self._subs.setdefault(topic, []).append(cb)
