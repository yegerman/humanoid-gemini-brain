"""Navigation: steer the GMT walk toward a target, halt on arrival.

GMT's walk clip steps at a fixed gait, so we steer by injecting a yaw-rate
(`controller.steer_yaw_rate`) computed from the error between the robot's actual travel
direction and the bearing to the target. On arrival we switch to the synthesized `stand`
motion to stop cleanly (scaling the walk's reference velocity alone doesn't halt it).
"""
from __future__ import annotations

import numpy as np

from synthesize import make_stand


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


class Navigator:
    def __init__(self, controller, walk_motion: str = "basic_walk.pkl",
                 arrive_tol: float = 0.7, kp: float = 1.6, obstacles=None) -> None:
        self.c = controller
        self.walk_motion = walk_motion
        self.arrive_tol = arrive_tol
        self.kp = kp
        self.obstacles = obstacles      # callable -> {label: (x, y)} of known objects, or None
        self._stand = make_stand()
        self.target = None
        self.arrived = False
        self._last_xy = None
        self._fallback_offset = -0.49  # rad (~-28 deg) yaw->travel offset, measured

    def go_to(self, x: float, y: float) -> None:
        self.target = np.array([float(x), float(y)])
        self.arrived = False
        self._last_xy = None
        self.c.set_motion(self.walk_motion)

    def update(self) -> dict:
        """Call once per control tick. Returns status dict for feedback/overlay."""
        p = self.c.get_proprio()
        xy = p.pos[:2]
        if self.target is None:
            return {"dist": 0.0, "arrived": True, "status": "idle"}
        d = self.target - xy
        dist = float(np.linalg.norm(d))

        if not self.arrived and dist < self.arrive_tol:
            self.arrived = True
            self.c.steer_yaw_rate = 0.0
            self.c.fwd_scale = 0.0
            self.c.set_motion(self._stand)
            return {"dist": dist, "arrived": True, "status": "arrived at target"}

        if self.arrived:
            return {"dist": dist, "arrived": True, "status": "standing at target"}

        # travel-direction heading (robust); fall back to yaw+offset when ~stationary
        if self._last_xy is not None:
            step = xy - self._last_xy
            if np.linalg.norm(step) > 1e-3:
                travel = np.arctan2(step[1], step[0])
            else:
                travel = p.yaw + self._fallback_offset
        else:
            travel = p.yaw + self._fallback_offset
        self._last_xy = xy.copy()

        bearing = np.arctan2(d[1], d[0])
        err = _wrap(bearing - travel)

        # Obstacle avoidance: repulsive steering away from known objects in the path ahead,
        # slow when close, hard-stop forward rather than stumbling into anything.
        fwd = 1.0
        if self.obstacles is not None:
            R = 1.3                                   # influence radius
            for label, (ox, oy) in dict(self.obstacles()).items():
                if any(w in label for w in ("red", "stage", "home", "circle")):
                    continue                           # flat / walkable markers
                ov = np.array([ox, oy]) - xy
                d_o = float(np.linalg.norm(ov))
                if d_o < 0.4 or d_o > R:
                    continue                           # <0.4 m = self / the carried box, not a wall
                if np.linalg.norm(np.array([ox, oy]) - self.target) < self.arrive_tol + 0.15:
                    continue                           # the obstacle IS (at) the destination
                ang = _wrap(float(np.arctan2(ov[1], ov[0])) - travel)
                if abs(ang) > 1.2:
                    continue                           # behind / far to the side
                side = np.sign(ang) if abs(ang) > 1e-3 else (np.sign(err) or 1.0)
                err += float(-side * 1.1 * (1.0 / d_o - 1.0 / R))   # steer away
                if abs(ang) < 0.7:                     # roughly dead ahead
                    if d_o < 0.45:
                        fwd = 0.0                      # stop, keep turning until clear
                    elif d_o < 0.8:
                        fwd = min(fwd, 0.6)            # ease off while skirting it

        self.c.steer_yaw_rate = float(np.clip(self.kp * err, -1.5, 1.5))
        self.c.fwd_scale = fwd
        return {"dist": dist, "arrived": False, "status": f"walking to target ({dist:.1f}m)"}
