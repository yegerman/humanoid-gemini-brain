"""Shared schemas for hosted and mock robot brains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


ActionName = Literal[
    "watch",
    "pick_and_place",
    "sort_all_matching",
    "stop",
    "ask_clarification",
]


@dataclass
class SceneObject:
    object_id: str
    label: str
    status: str
    color: str
    position: tuple[float, float, float]

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "label": self.label,
            "status": self.status,
            "color": self.color,
            "position_xyz": [round(v, 3) for v in self.position],
        }


@dataclass
class Decision:
    action: ActionName
    target_object: str = ""
    target_description: str = ""
    destination: str = "reject_bin"
    confidence: float = 0.0
    reasoning: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Decision":
        allowed = {"watch", "pick_and_place", "sort_all_matching", "stop", "ask_clarification"}
        action = str(raw.get("action", "watch")).strip()
        if action not in allowed:
            action = "watch"
        confidence = raw.get("confidence", 0.0)
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = 0.0
        return cls(
            action=action,  # type: ignore[arg-type]
            target_object=str(raw.get("target_object", "") or ""),
            target_description=str(raw.get("target_description", "") or ""),
            destination=str(raw.get("destination", "reject_bin") or "reject_bin"),
            confidence=confidence,
            reasoning=str(raw.get("reasoning", "") or ""),
        )

    def is_executable(self) -> bool:
        return self.action in {"pick_and_place", "sort_all_matching"} and bool(self.target_object)
