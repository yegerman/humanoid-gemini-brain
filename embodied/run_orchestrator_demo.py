"""ER-boss demo (M3.6): Gemini-ER orchestrates every command; 3.5 Flash authors new skills.

Same scene, controller, HUD and chat as the classic demo (`run_navigation_demo.py`), but the
brain is `orchestrator.OrchestratorBrain` — ER decides each command from the live camera +
spatial memory, and delegates unknown skills to the 3.5 sub-agent which synthesizes them on the
fly. The classic demo is unchanged; run whichever you like:

  classic (3.5 intent):   python embodied/run_navigation_demo.py
  ER-boss (this file):    python embodied/run_orchestrator_demo.py [--look-secs N]
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "controller"))

import cv2

# Reuse the classic demo's building blocks (build/Executor/skill registry/overlay/crash log).
import run_navigation_demo as base
from run_navigation_demo import build, SKILL_MOTIONS, _CRASH_LOG
from messaging import Feedback
from chat import Chat
from orchestrator import OrchestratorBrain
import overlay


def run_interactive(seconds: float, width: int = 1280, height: int = 720,
                    er_secs: float = 30.0) -> int:
    bus, c, percept, _classic_brain, nav, ex, scene, memory = build(width, height)
    # Swap in the ER orchestrator; it registers authored skills straight into SKILL_MOTIONS
    # so skill_motion() finds them. ER decisions AND ER scene-refresh are both throttled to
    # ~once per er_secs to save money; between them the cheap local/3.5 planner handles commands.
    brain = OrchestratorBrain(skill_registry=SKILL_MOTIONS, er_period_s=er_secs)
    look_secs = er_secs
    last_er = -10**9
    chat = Chat(); chat.start()
    print(f"BRAIN: Gemini-ER orchestrator (ER ~every {er_secs:.0f}s to save cost)"
          if brain._client else "BRAIN: offline fallback (no key)")

    import mujoco.viewer
    viewer = mujoco.viewer.launch_passive(c.model, c.data)
    viewer.cam.lookat[:] = [1.0, 0.0, 0.6]
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 5.0, 90.0, -20.0

    hud = "G1 HUD (ER-boss) - sees / thinks"
    cv2.namedWindow(hud, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(hud, 720, 540)

    render_every = 2
    max_ticks = int(seconds / c.control_dt) if seconds and seconds > 0 else None
    print("3D view: scroll = zoom, left-drag = orbit, right-drag = pan")
    print("ER decides each command; unknown skills are authored by 3.5. Type 'quit' to exit.")
    try:
        t = 0
        while True:
            if chat.quit.is_set() or not viewer.is_running():
                break
            if max_ticks is not None and t >= max_ticks:
                print("  [info] safety time cap reached.")
                break
            for _ in range(c.sim_decimation):
                c.step()
            cmd = chat.poll()
            if cmd:
                ego = percept.render_ego()                  # ER sees what the robot sees
                goal, plan = brain.plan(cmd, scene, memory, image=ego)
                ex.set(goal, plan)
                print(f"  [{plan.brain or '?'}] {goal.kind} {goal.target_xy or goal.skill} | {plan.reasoning}")
            ex.tick()
            if t % render_every == 0:
                viewer.sync()
                p = c.get_proprio()
                fb = Feedback(pos=tuple(float(x) for x in p.pos), yaw=p.yaw,
                              height=p.height, upright=p.upright)
                bus.publish("/feedback", fb)
                ego = percept.render_ego()
                if look_secs and ex.vision and ex.vision.er_available \
                        and ex.goal.kind not in ("go_to_visual", "look_at") \
                        and (c.counter - last_er) >= int(look_secs / c.sim_dt):
                    last_er = c.counter
                    ex._do_look()
                else:
                    ex.ambient_perceive(ego)
                frame = overlay.draw(ego, ex.goal, scene, ex.plan, fb)
                cv2.imshow(hud, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if cv2.getWindowProperty(hud, cv2.WND_PROP_VISIBLE) < 1:
                    break
            t += 1
    except Exception:
        traceback.print_exc()
        with open(_CRASH_LOG, "a") as f:
            f.write("\n--- Python exception (orchestrator) ---\n")
            traceback.print_exc(file=f)
        print(f"  [crash] traceback written to {_CRASH_LOG}")
    finally:
        try:
            viewer.close()
        except Exception:
            pass
        chat.stop()
        cv2.destroyAllWindows()
        for _ in range(3):
            cv2.waitKey(1)
        percept.close()
    print("bye.")
    import os
    os._exit(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="optional safety time cap in seconds; 0 = run until you quit")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--er-secs", type=float, default=30.0,
                    help="call Gemini-ER (see + decide) at most once per N seconds to save cost; "
                         "between calls the cheap local/3.5 planner handles commands. 0 = ER every command.")
    a = ap.parse_args()
    raise SystemExit(run_interactive(a.seconds, a.width, a.height, a.er_secs))
