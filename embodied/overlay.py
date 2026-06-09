"""Overlay HUD on the camera frame: what it sees, what it's thinking, proprio.

Draws onto the RGB frame (returns BGR for cv2 display), mirroring the annotate() style
from g1_direct_factory_demo.py.
"""
from __future__ import annotations

import cv2
import numpy as np

from messaging import Feedback, Plan, SceneView, Goal


def draw(rgb: np.ndarray, goal: Goal, scene: SceneView, plan: Plan, fb: Feedback) -> np.ndarray:
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = out.shape[:2]
    s = h / 480.0  # scale HUD with frame height (design was 480p)

    # top banner: what it SEES + current command + remembered objects
    mem = scene.targets or {}
    _band(out, 0, int(88 * s))
    _put(out, f"SEES: {scene.caption[:70] or '(scene)'}", int(22 * s), (140, 220, 255), s)
    _put(out, f"CMD : {goal.text[:70] or '(idle)'}", int(46 * s), (200, 235, 200), s)
    labels = ", ".join(list(mem.keys())[:4])
    _put(out, f"MEM : {len(mem)} objects [{labels[:60]}]", int(70 * s), (210, 190, 255), s)

    # bottom banner: what it's THINKING + proprio HUD
    y0 = h - int(96 * s)
    _band(out, y0, h)
    _put(out, f"THINKS: {(plan.current or plan.reasoning)[:74]}", y0 + int(22 * s), (255, 220, 130), s)
    step = " > ".join(plan.steps[:3]) if plan.steps else "-"
    _put(out, f"PLAN  : {step[:74]}", y0 + int(46 * s), (230, 230, 230), s)
    up = "UP" if fb.upright else "DOWN"
    _put(out, f"POS ({fb.pos[0]:+.2f},{fb.pos[1]:+.2f}) h={fb.height:.2f} yaw={np.degrees(fb.yaw):+.0f} [{up}]",
         y0 + int(70 * s), (180, 230, 180) if fb.upright else (120, 120, 255), s)
    # Which brain decided (ER-boss demo only; classic demo leaves plan.brain unset -> no line).
    if getattr(plan, "brain", ""):
        _put(out, f"BRAIN: {plan.brain}", h - int(8 * s), (255, 180, 120), s)
    return out


def _band(img, y0, y1):
    ov = img.copy()
    cv2.rectangle(ov, (0, y0), (img.shape[1], y1), (18, 20, 24), -1)
    cv2.addWeighted(ov, 0.62, img, 0.38, 0, img)


def _put(img, text, y, color, s=1.0):
    cv2.putText(img, text, (int(14 * s), y), cv2.FONT_HERSHEY_SIMPLEX,
                0.55 * s, color, max(1, int(round(s))), cv2.LINE_AA)
