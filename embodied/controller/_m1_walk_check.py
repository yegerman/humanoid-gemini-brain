"""Headless M1 check: confirm the vendored policy walks (no viewer)."""
from __future__ import annotations

import numpy as np

from walk_controller import WalkController


def main() -> int:
    c = WalkController()
    c.set_cmd_vel(0.3, 0.0, 0.0)  # walk forward
    start = c.get_proprio().pos.copy()
    heights = []
    seconds = 6.0
    steps = int(seconds / c.simulation_dt)
    for i in range(steps):
        c.step()
        if i % 50 == 0:
            heights.append(c.get_proprio().height)
    end = c.get_proprio()
    from walk_controller import get_gravity_orientation
    gz = float(get_gravity_orientation(end.quat)[2])
    dx = float(end.pos[0] - start[0])
    print(f"start xy = {start[:2]}")
    print(f"end   xy = {end.pos[:2]}  height={end.height:.3f}  upright={end.upright} gz={gz:.3f}")
    print(f"forward distance dx = {dx:.3f} m over {seconds:.0f}s")
    print(f"min height = {min(heights):.3f}  max height = {max(heights):.3f}")
    ok = end.upright and end.height > 0.5 and dx > 0.3
    print("RESULT:", "WALKS-OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
