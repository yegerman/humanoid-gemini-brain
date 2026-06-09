"""Confirm set_cmd_vel is live: forward -> turn -> stop."""
import numpy as np
from walk_controller import WalkController


def yaw(c):
    w, x, y, z = c.data.qpos[3:7]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


c = WalkController()
c.set_cmd_vel(0.3, 0, 0)
for _ in range(int(2 / c.simulation_dt)):
    c.step()
p1, y1 = c.get_proprio(), yaw(c)

c.set_cmd_vel(0.0, 0, 0.6)
for _ in range(int(2.5 / c.simulation_dt)):
    c.step()
p2, y2 = c.get_proprio(), yaw(c)

c.set_cmd_vel(0, 0, 0)
for _ in range(int(1 / c.simulation_dt)):
    c.step()
p3 = c.get_proprio()

print(f"after FWD : xy={p1.pos[:2]} yaw={np.degrees(y1):.1f} upright={p1.upright}")
print(f"after TURN: xy={p2.pos[:2]} yaw={np.degrees(y2):.1f} upright={p2.upright} dyaw={np.degrees(y2-y1):.1f}")
print(f"after STOP: xy={p3.pos[:2]} height={p3.height:.3f} upright={p3.upright}")
print("LIVE-CMD-OK" if abs(np.degrees(y2 - y1)) > 20 and p3.upright else "CHECK")
