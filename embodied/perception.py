"""Perception: render the robot's camera + read proprioception, publish to the bus.

Publishes /camera (CameraFrame) and /feedback (Feedback) every tick. The camera frame is
the input the Gemini-ER planner reasons over; proprio is the cheap success-check signal.
"""
from __future__ import annotations

import numpy as np
import mujoco

from messaging import Bus, CameraFrame, Feedback


class Perception:
    def __init__(self, controller, bus: Bus, camera: str = "chase",
                 width: int = 640, height: int = 480) -> None:
        self.c = controller
        self.bus = bus
        self.camera = camera
        self.free_cam = None   # optional mujoco.MjvCamera; overrides `camera` when set (mouse zoom/orbit)
        self.renderer = mujoco.Renderer(controller.model, height=height, width=width)

    def render(self) -> np.ndarray:
        cam = self.free_cam if self.free_cam is not None else self.camera
        self.renderer.update_scene(self.c.data, camera=cam)
        return self.renderer.render()  # RGB

    def render_ego(self) -> np.ndarray:
        """Egocentric onboard-camera frame (what the robot actually sees) for the vision brain."""
        if not hasattr(self, "_ego"):
            import mujoco
            self._ego = mujoco.Renderer(self.c.model, height=480, width=640)
        self._ego.update_scene(self.c.data, camera="ego")
        return self._ego.render()

    def tick(self) -> tuple[np.ndarray, Feedback]:
        rgb = self.render()
        p = self.c.get_proprio()
        fb = Feedback(pos=tuple(float(x) for x in p.pos), yaw=p.yaw,
                      height=p.height, upright=p.upright)
        self.bus.publish("/camera", CameraFrame(rgb=rgb, camera=self.camera))
        self.bus.publish("/feedback", fb)
        return rgb, fb

    def close(self) -> None:
        self.renderer.close()
