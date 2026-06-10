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
import time
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

MODELS = ["gemini-robotics-er-1.6-preview", "gemini-2.5-flash", "gemini-2.0-flash"]
ER_COOLDOWN_S = 60.0   # after a 429, rest a model this long then retry (paid recovers per-minute)

# HSV color ranges (OpenCV H is 0-180) matching the scene props + red stage disk.
COLOR_RANGES = {
    "red": [((0, 90, 60), (12, 255, 255)), ((168, 90, 60), (180, 255, 255))],
    "orange": [((13, 120, 80), (24, 255, 255))],
    "yellow": [((25, 90, 80), (35, 255, 255))],
    "green": [((36, 70, 50), (84, 255, 255))],
    "cyan": [((85, 120, 80), (97, 255, 255))],    # the carry box
    "blue": [((98, 130, 60), (130, 255, 255))],   # high S/V so the gray-blue floor tiles don't match
    "purple": [((131, 60, 50), (152, 255, 255))],
    "magenta": [((153, 80, 70), (167, 255, 255))],   # the magenta carry box
}
MAX_BLOB_FRAC = 0.40   # reject blobs bigger than this fraction of the frame (floor/walls/background)

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
        self._cooldown: dict[str, float] = {}  # model -> monotonic time until it may retry
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if key:
            try:
                from google import genai
                from google.genai import types
                # Hard request timeout so a hung vision call can't freeze the loop (local CV covers).
                self._client = genai.Client(api_key=key,
                                            http_options=types.HttpOptions(timeout=15_000))
                self._types = types
            except Exception:
                self._client = None

    @property
    def available(self) -> bool:
        # Always available: local OpenCV detection works even with no API / exhausted quota.
        return True

    @property
    def er_available(self) -> bool:
        now = time.monotonic()
        return self._client is not None and any(self._cooldown.get(m, 0.0) <= now for m in self.models)

    def look(self, rgb: np.ndarray) -> dict | None:
        """rgb: HxWx3 RGB. Tries Gemini-ER for a rich caption; falls back to local
        OpenCV red-disk detection (free, unlimited) so the robot can always see the circle."""
        er = self._look_er(rgb) if self.er_available else None
        if er is not None:
            return self._refine_with_local(er, rgb)
        return self._look_local(rgb)

    def _refine_with_local(self, er: dict, rgb: np.ndarray) -> dict:
        """Keep ER's caption/labels but use the LOCAL detector's pixel-accurate geometry
        (cx, by base-pixel, size) for color-matched objects. ER's coarse cy/size (and missing
        'by') put ground-projected positions ~1-2m off; local CV is exact and free."""
        loc = self._look_local(rgb)
        by_color = {o["color"]: o for o in loc.get("objects", [])}
        for o in er.get("objects", []) or []:
            color = next((c for c in COLOR_RANGES if c in str(o.get("label", "")).lower()), None)
            m = by_color.get(color)
            if m:
                o["cx"], o["cy"], o["by"], o["size"] = m["cx"], m["cy"], m["by"], m["size"]
        if loc["red_disk"]["visible"]:
            er["red_disk"] = loc["red_disk"]
            er["red_disk"]["by"] = by_color.get("red", {}).get("by", loc["red_disk"]["cy"])
        return er

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
        now = time.monotonic()
        for model in self.models:
            if self._cooldown.get(model, 0.0) > now:
                continue   # still resting after a recent 429
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
                    self._cooldown[model] = time.monotonic() + ER_COOLDOWN_S  # retry later
                continue
        return None

    def _look_local(self, rgb: np.ndarray) -> dict:
        """Free multi-color HSV detection — names the colored props in view (no API), so the
        live caption + memory track the camera even when ER quota is unavailable. Shape is a
        coarse aspect-ratio guess; the red one is also returned as `red_disk` for servoing."""
        h, w = rgb.shape[:2]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        objects = []
        red = {"visible": False, "cx": 0.0, "cy": 0.0, "size": 0.0}
        for color, ranges in COLOR_RANGES.items():
            mask = None
            for lo, hi in ranges:
                m = cv2.inRange(hsv, lo, hi)
                mask = m if mask is None else (mask | m)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area < 80 or area > MAX_BLOB_FRAC * w * h:
                continue   # too small (noise) or too big (floor/background)
            x, y, bw, bh = cv2.boundingRect(c)
            cx = ((x + bw / 2) / w) * 2 - 1
            cy = ((y + bh / 2) / h) * 2 - 1
            # Floor-contact pixel for ground projection: tall props meet the floor at the bbox
            # BOTTOM; the red disk lies FLAT on the floor, so its CENTER pixel is the disk center.
            by = cy if color == "red" else ((y + bh) / h) * 2 - 1
            size = float(bw / w)
            shape = self._guess_shape(color, bw, bh)
            objects.append({"label": f"{color} {shape}", "color": color, "cx": float(cx),
                            "cy": float(cy), "by": float(by), "size": size,
                            "_area": float(area)})
            if color == "red":
                red = {"visible": True, "cx": float(cx), "cy": float(cy), "size": size}
        objects.sort(key=lambda o: o["_area"], reverse=True)
        caption = self._caption(objects)
        for o in objects:
            o.pop("_area", None)
        return {"caption": caption, "red_disk": red, "objects": objects, "model_used": "local-cv"}

    @staticmethod
    def _guess_shape(color: str, bw: int, bh: int) -> str:
        if color == "red":
            return "circle"          # the stage disk
        if color in ("cyan", "magenta"):
            return "box"             # the carry boxes
        aspect = bw / max(1, bh)
        if aspect < 0.7:
            return "pillar"          # tall and narrow
        if aspect > 1.4:
            return "box"             # wide
        return "sphere"

    @staticmethod
    def _caption(objects: list) -> str:
        if not objects:
            return "Onboard view: a checkered floor; nothing notable in sight."
        parts = []
        for o in objects[:3]:
            cx = o["cx"]
            side = "ahead" if abs(cx) < 0.2 else ("to the right" if cx > 0 else "to the left")
            parts.append(f"a {o['label']} {side}")
        return "Onboard view: " + ", ".join(parts) + "."


if __name__ == "__main__":
    import sys
    img = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2RGB)
    print(json.dumps(VisionBrain().look(img), indent=2))
