"""Gemini hosted vision brain adapter."""

from __future__ import annotations

import base64
import json
import os
import re
from typing import Any

import cv2
import numpy as np

from .schemas import Decision


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class GeminiBrain:
    """Calls Gemini with a camera frame and instruction, returning a Decision."""

    def __init__(self, model: str = "gemini-2.5-flash") -> None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY before using --brain gemini.")
        from google import genai
        from google.genai import types

        self._client = genai.Client(api_key=api_key)
        self._types = types
        self._model = model

    def decide(
        self,
        frame_bgr: np.ndarray,
        instruction: str,
        scene_hint: dict[str, Any],
    ) -> Decision:
        ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return Decision(action="watch", reasoning="Could not encode camera frame.")

        prompt = self._build_prompt(instruction, scene_hint)
        image_part = self._types.Part.from_bytes(
            data=encoded.tobytes(),
            mime_type="image/jpeg",
        )
        response = self._client.models.generate_content(
            model=self._model,
            contents=[prompt, image_part],
            config=self._types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        return self._parse_response(getattr(response, "text", "") or "")

    def _build_prompt(self, instruction: str, scene_hint: dict[str, Any]) -> str:
        return (
            "You are the visual reasoning brain for a MuJoCo factory robot.\n"
            "Look at the camera frame and the scene hint. Choose the next safe robot action.\n"
            "Return ONLY valid JSON with this exact shape:\n"
            "{\n"
            '  "action": "watch|pick_and_place|sort_all_matching|stop|ask_clarification",\n'
            '  "target_object": "object_id from scene hint, or empty",\n'
            '  "target_description": "short visual description",\n'
            '  "destination": "reject_bin|good_bin",\n'
            '  "confidence": 0.0,\n'
            '  "reasoning": "one concise sentence"\n'
            "}\n\n"
            f"User instruction: {instruction}\n\n"
            "Scene hint JSON:\n"
            f"{json.dumps(scene_hint, indent=2)}\n\n"
            "Rules:\n"
            "- Defective parts should go to reject_bin unless the user says otherwise.\n"
            "- Good parts should be ignored unless the user asks to sort them.\n"
            "- If the user asks for all matching objects, choose ONE visible matching object as the next target.\n"
            "- For executable actions, target_object must be exactly one object_id from the scene hint.\n"
            "- Prefer action=pick_and_place for the next object to move.\n"
            "- If no matching visible object exists, action must be watch.\n"
            "- Use only object IDs that appear in the scene hint."
        )

    def _parse_response(self, text: str) -> Decision:
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            match = JSON_RE.search(text)
            if not match:
                return Decision(action="watch", reasoning=f"Gemini returned non-JSON: {text[:120]}")
            try:
                raw = json.loads(match.group(0))
            except json.JSONDecodeError:
                return Decision(action="watch", reasoning="Gemini JSON could not be parsed.")
        if not isinstance(raw, dict):
            return Decision(action="watch", reasoning="Gemini did not return a JSON object.")
        return Decision.from_dict(raw)
