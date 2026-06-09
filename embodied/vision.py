"""Vision brain: the robot's eyes, via Gemini Robotics-ER 1.6.

Sends the onboard camera frame to gemini-robotics-er-1.6-preview (with a fallback chain to
2.5-flash / 2.0-flash if a model is rate-limited) and returns a one-line caption plus the
located red stage disk in normalized image coords. Used for:
  * Gate C — "what do you see": real scene caption.
  * Gate D — "go to the red circle": visual servoing toward the detected disk.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

MODELS = ["gemini-robotics-er-1.6-preview", "gemini-2.5-flash", "gemini-2.0-flash"]

PROMPT = """You are the eyes of a humanoid robot, looking forward from its head.
1) Describe what you see in ONE short sentence.
2) Locate the RED circular disk on the floor, if visible.
3) List the other distinct objects you see (boxes, balls, pillars, posts, props), each with
   a short color+shape label (e.g. "green sphere", "blue crate", "yellow pillar").
Coordinates: cx in -1..1 (left..right of image center), cy in -1..1 (top..bottom),
size = object width as a fraction of image width (0..1).
Return ONLY JSON:
{"caption":"...",
 "red_disk":{"visible":true|false,"cx":0,"cy":0,"size":0},
 "objects":[{"label":"green sphere","cx":0.3,"cy":0.1,"size":0.12}]}
If the red disk is not visible set visible=false and cx/cy/size to 0. If you see no other
objects return an empty objects list."""


class VisionBrain:
    def __init__(self, models=MODELS) -> None:
        self.models = list(models)
        self._client = None
        self._dead: set[str] = set()  # models that returned quota errors this session
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if key:
            try:
                from google import genai
                from google.genai import types
                self._client = genai.Client(api_key=key)
                self._types = types
            except Exception:
                self._client = None

    @property
    def available(self) -> bool:
        # Always available: local OpenCV detection works even with no API / exhausted quota.
        return True

    @property
    def er_available(self) -> bool:
        return self._client is not None and any(m not in self._dead for m in self.models)

    def look(self, rgb: np.ndarray) -> dict | None:
        """rgb: HxWx3 RGB. Tries Gemini-ER for a rich caption; falls back to local
        OpenCV red-disk detection (free, unlimited) so the robot can always see the circle."""
        er = self._look_er(rgb) if self.er_available else None
        if er is not None:
            return er
        return self._look_local(rgb)

    def quick_look(self, rgb: np.ndarray) -> dict:
        """Free, unlimited LOCAL caption (no API call) for the live HUD — keeps the SEES text
        tracking the camera between explicit ER look commands without burning vision quota."""
        return self._look_local(rgb)

    def _look_er(self, rgb: np.ndarray) -> dict | None:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return None
        part = self._types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg")
        for model in self.models:
            if model in self._dead:
                continue
            try:
                resp = self._client.models.generate_content(
                    model=model, contents=[PROMPT, part],
                    config=self._types.GenerateContentConfig(temperature=0.0),
                )
                text = getattr(resp, "text", "") or ""
                m = JSON_RE.search(text)
                if m:
                    data = json.loads(m.group(0))
                    data["model_used"] = model
                    return data
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    self._dead.add(model)
                continue
        return None

    def _look_local(self, rgb: np.ndarray) -> dict:
        """Local red-disk detection via HSV color thresholding on the onboard frame."""
        h, w = rgb.shape[:2]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, (0, 90, 60), (12, 255, 255)) | \
            cv2.inRange(hsv, (168, 90, 60), (180, 255, 255))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red = {"visible": False, "cx": 0.0, "cy": 0.0, "size": 0.0}
        caption = "Onboard view: a checkered floor; no red circle in sight."
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > 60:
                x, y, bw, bh = cv2.boundingRect(c)
                cx = ((x + bw / 2) / w) * 2 - 1
                cy = ((y + bh / 2) / h) * 2 - 1
                red = {"visible": True, "cx": float(cx), "cy": float(cy), "size": float(bw / w)}
                side = "ahead" if abs(cx) < 0.2 else ("to the right" if cx > 0 else "to the left")
                caption = f"Onboard view: a red circular disk on the checkered floor, {side}."
        # Local CV only knows the red disk; mirror it into objects for memory consistency.
        objects = [{"label": "red circle", "cx": red["cx"], "cy": red["cy"],
                    "size": red["size"]}] if red["visible"] else []
        return {"caption": caption, "red_disk": red, "objects": objects,
                "model_used": "local-cv"}


if __name__ == "__main__":
    import sys
    img = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2RGB)
    print(json.dumps(VisionBrain().look(img), indent=2))
