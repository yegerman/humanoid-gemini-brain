"""Deterministic offline brain for demo and testing."""

from __future__ import annotations

from typing import Any

import numpy as np

from .schemas import Decision


class MockBrain:
    """Local stand-in for Gemini.

    It reads the same instruction and scene hint as Gemini, then chooses a
    reasonable target deterministically. This keeps the full demo runnable
    without internet or API keys.
    """

    def decide(
        self,
        frame_bgr: np.ndarray,
        instruction: str,
        scene_hint: dict[str, Any],
    ) -> Decision:
        objects = [obj for obj in scene_hint.get("objects", []) if obj.get("object_id")]
        text = instruction.lower()

        candidates = objects
        if "red" in text:
            candidates = [obj for obj in candidates if obj.get("color") == "red"]
        elif "dark" in text or "gray" in text or "grey" in text:
            candidates = [obj for obj in candidates if obj.get("color") in {"dark", "gray", "grey"}]

        if "defect" in text or "bad" in text or "reject" in text:
            candidates = [obj for obj in candidates if obj.get("status") == "defective"]

        if "green" in text and "ignore" not in text:
            candidates = [obj for obj in objects if obj.get("color") == "green"]

        if not candidates:
            return Decision(
                action="watch",
                confidence=0.8,
                reasoning="No matching object is visible for the instruction.",
            )

        target = candidates[0]
        destination = "reject_bin" if target.get("status") == "defective" else "good_bin"
        return Decision(
            action="pick_and_place",
            target_object=target["object_id"],
            target_description=f"{target.get('color')} {target.get('label')} ({target.get('status')})",
            destination=destination,
            confidence=0.95,
            reasoning=f"Mock brain selected {target['object_id']} because it matches: {instruction}",
        )
