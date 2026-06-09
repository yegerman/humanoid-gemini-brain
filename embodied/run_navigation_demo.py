"""Embodied navigation demo — chat-driven whole-body G1 (M3.5).

Type commands; Gemini parses intent (grounded in skills + spatial memory + vision); the robot
walks to the stage / a remembered object, performs a gesture, stands up, or searches for an
unseen target — all whole-body via GMT, with a live overlay (sees / mem / thinks / proprio).
The Executor learns every detected object into a shared SpatialMemory so the planner never
invents positions. See DESIGN.md for the perception -> memory -> planner closed loop.

  interactive:  python embodied/run_navigation_demo.py
  scripted gate test (headless, saves frames):
                python embodied/run_navigation_demo.py --script
"""
from __future__ import annotations

import argparse
import faulthandler
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "controller"))

# Crash diagnostics: native segfaults (e.g. OpenGL/renderer) leave no Python traceback and
# no MuJoCo log, so dump a C-level stack trace to _crash.log on any fatal fault.
_CRASH_LOG = Path(__file__).resolve().parent / "_crash.log"
try:
    _crash_fp = open(_CRASH_LOG, "w")
    faulthandler.enable(file=_crash_fp, all_threads=True)
except Exception:
    faulthandler.enable(all_threads=True)

import cv2
import numpy as np

from controller.gmt_controller import GMTController, STAGE_SCENE, MOTIONS
from messaging import Bus, Goal, Plan, SceneView, Feedback
from perception import Perception
from planner import Brain
from nav import Navigator
from chat import Chat
from vision import VisionBrain
from memory import SpatialMemory
import overlay
import synthesize

SKILL_MOTIONS = synthesize.make_all()  # {"stand":path, "raise_right_hand":path}


def skill_motion(skill: str) -> str:
    if skill in SKILL_MOTIONS:
        return SKILL_MOTIONS[skill]
    # clip fallbacks
    clip = MOTIONS / f"{skill}.pkl"
    return str(clip) if clip.exists() else SKILL_MOTIONS["stand"]


class Executor:
    """Applies the current goal each control tick and produces a status string."""

    def __init__(self, controller, nav: Navigator, vision, percept, scene, memory=None) -> None:
        self.c = controller
        self.nav = nav
        self.vision = vision
        self.percept = percept
        self.scene = scene
        self.memory = memory
        self.goal = Goal()
        self.plan = Plan(current="idle")
        self.turn_end = None
        self._vis_next = 0          # counter at which to next call the vision brain
        self._vis = None            # last vision result
        self._arrived_v = False
        self._was_close = False

    def set(self, goal: Goal, plan: Plan) -> None:
        self.goal, self.plan = goal, plan
        self.turn_end = None
        if goal.kind == "go_to" and goal.target_xy:
            self.nav.go_to(*goal.target_xy)
        elif goal.kind == "look_at":
            self._begin_look_at(goal)
        elif goal.kind == "go_to_visual":
            self.c.set_motion(self.nav.walk_motion)
            self.c.fwd_scale = 0.0
            self.c.steer_yaw_rate = 0.0
            self._vis_next = self.c.counter
            self._vis = None
            self._arrived_v = False
            self._visual_phase = "acquire"   # acquire (look) -> nav (odometry walk to estimate)
        elif goal.kind == "look":
            self.c.steer_yaw_rate = 0.0
            self.c.fwd_scale = 0.0
            self.c.set_motion(skill_motion("stand"))
            self._do_look()
        elif goal.kind == "skill" and goal.skill == "stand_up":
            # Recover-to-stand: reset to a stable standing pose, then hold the stand clip.
            self.c.recover_to_stand()
            self.c.steer_yaw_rate = 0.0
            self.c.fwd_scale = 0.0
            self.c.set_motion(skill_motion("stand_up"))
            self.plan.current = "standing up (recovered to a stable pose)"
        elif goal.kind == "skill" and goal.skill in ("turn_left", "turn_right"):
            self.c.set_motion(self.nav.walk_motion)
            self.c.fwd_scale = 0.15
            self.c.steer_yaw_rate = 0.8 if goal.skill == "turn_left" else -0.8
            self.turn_end = self.c.counter + int(2.5 / self.c.sim_dt)
        elif goal.kind == "skill":
            self.c.steer_yaw_rate = 0.0
            self.c.fwd_scale = 1.0
            self.c.set_motion(skill_motion(goal.skill), force=True)
        else:
            self.c.steer_yaw_rate = 0.0
            self.c.fwd_scale = 0.0
            self.c.set_motion(skill_motion("stand"))

    def _do_look(self) -> str:
        r = self.vision.look(self.percept.render_ego()) if self.vision else None
        if r:
            self.scene.caption = r.get("caption", "(no caption)")
            self._learn(r)
            self.plan.current = "I see: " + self.scene.caption[:60]
        else:
            self.scene.caption = "(vision unavailable - quota?)"
            self.plan.current = "vision unavailable"
        print("  vision:", self.scene.caption)
        return self.plan.current

    def _learn(self, r: dict) -> None:
        """Record every detected object into spatial memory + scene.targets (label -> world xy)."""
        if not r or self.memory is None:
            return
        pr = self.c.get_proprio()
        dets = list(r.get("objects", []))
        rd = r.get("red_disk", {})
        if rd.get("visible") and not any((d.get("label") or "").strip().lower() == "red circle" for d in dets):
            dets.append({"label": "red circle", "cx": rd.get("cx", 0.0),
                         "cy": rd.get("cy", 0.0), "size": rd.get("size", 0.0)})
        for d in dets:
            label = (d.get("label") or "").strip().lower()
            if not label or float(d.get("size", 0.0)) <= 0.0:
                continue
            tx, ty = self._estimate_target(pr, d)
            self.memory.update(label, (tx, ty), self.c.counter)
            self.scene.targets[label] = (tx, ty)

    def _begin_look_at(self, goal: Goal) -> None:
        """Recall a remembered object and turn in place to face it, then look (Voyager recall:
        'I saw the red circle earlier; now look back at it'). Unknown -> behave like plain look."""
        import math
        rec = self.memory.recall(goal.target_name) if (self.memory and goal.target_name) else None
        self.c.fwd_scale = 0.0
        if rec is None:
            self.plan.current = f"haven't seen a {goal.target_name} yet — just looking"
            self.c.set_motion(skill_motion("stand"))
            self.c.steer_yaw_rate = 0.0
            self._do_look()
            self._look_at_pending = False
            return
        label, (tx, ty) = rec
        pr = self.c.get_proprio()
        desired = math.atan2(ty - pr.pos[1], tx - pr.pos[0])
        dyaw = math.atan2(math.sin(desired - pr.yaw), math.cos(desired - pr.yaw))
        self.c.set_motion(self.nav.walk_motion)
        self.c.steer_yaw_rate = 0.8 if dyaw > 0 else -0.8
        self.turn_end = self.c.counter + int(min(2.6, abs(dyaw) / 0.8) / self.c.sim_dt)
        self._look_at_pending = True
        self.plan.current = f"turning to face the {label}"

    def _servo(self) -> str:
        # Phase 1 — acquire: look until the red circle is seen, estimate its WORLD position
        # from bearing (cx) + apparent size (distance), then hand off to the odometry nav.
        if self._visual_phase == "acquire":
            self.c.fwd_scale = 0.0
            if self.c.counter >= self._vis_next and self.vision and self.vision.available:
                r = self.vision.look(self.percept.render_ego())
                self._vis_next = self.c.counter + int(1.0 / self.c.sim_dt)
                if r is not None and r.get("caption"):
                    self.scene.caption = r["caption"]
                if r is not None:
                    self._learn(r)
                rd = (r or {}).get("red_disk", {})
                if rd.get("visible"):
                    tx, ty = self._estimate_target(self.c.get_proprio(), rd)
                    self.nav.go_to(tx, ty)
                    self._visual_phase = "nav"
                    self.plan.current = f"saw the red circle, walking to (~{tx:.1f},{ty:.1f})"
                    self.c.steer_yaw_rate = 0.0
                    return self.plan.current
            self.c.steer_yaw_rate = 0.4   # rotate to bring the disk into view
            self.plan.current = "looking for the red circle"
            return self.plan.current
        # Phase 2 — navigate to the estimated world position (reuses the proven Navigator).
        st = self.nav.update()
        self.plan.current = st["status"]
        if st["arrived"]:
            self._arrived_v = True
        return self.plan.current

    def _estimate_target(self, pr, rd) -> tuple:
        """One detection -> disk world (x,y): bearing from cx, range from apparent size."""
        import math
        w, h = 640, 480
        fovy = math.radians(78.0)
        fovx = 2 * math.atan(math.tan(fovy / 2) * w / h)
        cx = float(rd.get("cx", 0.0))
        size = max(0.05, float(rd.get("size", 0.15)))
        angle = math.atan(cx * math.tan(fovx / 2))   # horizontal bearing offset (cam frame)
        rng = 0.375 / size                            # calibrated camera->disk ground range
        heading = pr.yaw + angle                      # body +x (camera forward) is along yaw
        tx = float(pr.pos[0] + rng * math.cos(heading))
        ty = float(pr.pos[1] + rng * math.sin(heading))
        return tx, ty

    def _arrive(self) -> str:
        self._arrived_v = True
        self.c.steer_yaw_rate = 0.0
        self.c.fwd_scale = 0.0
        self.c.set_motion(skill_motion("stand"))
        self.plan.current = "arrived at the red circle"
        return self.plan.current

    def tick(self) -> str:
        if self.goal.kind == "go_to":
            self.plan.current = self.nav.update()["status"]
            return self.plan.current
        if self.goal.kind == "go_to_visual":
            return self._servo()
        if self.goal.kind == "look":
            return self.plan.current
        if self.goal.kind == "look_at":
            if self.turn_end is not None and self.c.counter >= self.turn_end:
                self.c.steer_yaw_rate = 0.0
                self.c.fwd_scale = 0.0
                self.c.set_motion(skill_motion("stand"))
                self.turn_end = None
                if getattr(self, "_look_at_pending", False):
                    self._look_at_pending = False
                    self._do_look()   # re-capture the object now that we face it
            return self.plan.current
        if self.goal.kind == "skill":
            if self.turn_end is not None:
                if self.c.counter >= self.turn_end:
                    self.c.steer_yaw_rate = 0.0
                    self.c.fwd_scale = 0.0
                    self.c.set_motion(skill_motion("stand"))
                    self.turn_end = None
                    self.plan.current = "done turning"
                else:
                    self.plan.current = f"{self.goal.skill.replace('_', ' ')}"
                return self.plan.current
            self.plan.current = f"performing {self.goal.skill}"
            return self.plan.current
        return "idle"


def build(width: int = 1280, height: int = 720):
    bus = Bus()
    c = GMTController(motion="basic_walk.pkl", scene=str(STAGE_SCENE))
    percept = Perception(c, bus, camera="chase", width=width, height=height)
    brain = Brain()
    vision = VisionBrain()
    nav = Navigator(c)
    scene = SceneView(caption="(robot's onboard view)", targets={"stage center": (2.5, 0.0)})
    memory = SpatialMemory()   # one shared memory: Executor writes detections, Brain reads for grounding
    ex = Executor(c, nav, vision, percept, scene, memory)
    return bus, c, percept, brain, nav, ex, scene, memory


def run_interactive(seconds: float, width: int = 1280, height: int = 720) -> int:
    bus, c, percept, brain, nav, ex, scene, memory = build(width, height)
    chat = Chat(); chat.start()
    print("LLM intent:", "Gemini" if brain._client else "offline fallback")

    # 3D view = MuJoCo's NATIVE passive viewer: rock-solid built-in mouse zoom (scroll),
    # orbit (left-drag), pan (right-drag) — no flaky OpenCV wheel handling.
    import mujoco.viewer
    from messaging import Feedback
    viewer = mujoco.viewer.launch_passive(c.model, c.data)
    # Start the free camera looking at the robot/stage from a nice angle.
    viewer.cam.lookat[:] = [1.0, 0.0, 0.6]
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 5.0, 90.0, -20.0

    # HUD window = the robot's onboard ("eyes") view + what it SEES / THINKS / its POSITION.
    hud = "G1 HUD - sees / thinks"
    cv2.namedWindow(hud, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(hud, 720, 540)

    render_every = 2  # refresh views every N control ticks
    # Runs until you quit: type quit/exit/q, press q in the HUD, or close either window.
    max_ticks = int(seconds / c.control_dt) if seconds and seconds > 0 else None
    print("3D view: scroll = zoom, left-drag = orbit, right-drag = pan (native MuJoCo viewer)")
    print("running until you quit (type 'quit'/'exit'/'q', press q, or close a window)"
          + (f"; safety cap {seconds:.0f}s" if max_ticks else ""))
    try:
        t = 0
        while True:
            if chat.quit.is_set() or not viewer.is_running():
                break
            if max_ticks is not None and t >= max_ticks:
                print("  [info] safety time cap reached.")
                break
            for _ in range(c.sim_decimation):   # one 50 Hz control tick
                c.step()
            cmd = chat.poll()
            if cmd:
                goal, plan = brain.plan(cmd, scene, memory)
                ex.set(goal, plan)
                print(f"  brain: {goal.kind} {goal.target_xy or goal.skill} | {plan.reasoning}")
            ex.tick()
            if t % render_every == 0:
                viewer.sync()                       # refresh the interactive 3D view
                p = c.get_proprio()
                fb = Feedback(pos=tuple(float(x) for x in p.pos), yaw=p.yaw,
                              height=p.height, upright=p.upright)
                bus.publish("/feedback", fb)
                ego = percept.render_ego()          # robot's-eye view
                frame = overlay.draw(ego, ex.goal, scene, ex.plan, fb)
                cv2.imshow(hud, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if cv2.getWindowProperty(hud, cv2.WND_PROP_VISIBLE) < 1:
                    break
            t += 1
    except Exception:
        # A Python-level error in the loop: log full traceback so it isn't lost when the
        # window closes, then keep going to the cleanup below.
        traceback.print_exc()
        with open(_CRASH_LOG, "a") as f:
            f.write("\n--- Python exception ---\n")
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
    # The chat thread is blocked on input() and GL/highgui can keep the process
    # alive on Windows; force a clean, immediate exit.
    import os
    os._exit(0)


def run_script() -> int:
    """Headless: run the two acceptance gates, save frames, print pass/fail."""
    bus, c, percept, brain, nav, ex, scene, memory = build()
    out = Path(__file__).resolve().parent

    def execute(cmd, seconds):
        goal, plan = brain.plan(cmd, scene, memory)
        ex.set(goal, plan)
        print(f"\n>>> {cmd!r} -> {goal.kind} {goal.target_xy or goal.skill}")
        for i in range(int(seconds / c.sim_dt)):
            c.step()
            if i % c.sim_decimation == 0:
                ex.tick()
        return ex.goal

    # Gate A
    execute("go to the center of the stage", 9.0)
    p = c.get_proprio()
    distA = float(np.linalg.norm(np.array([2.5, 0.0]) - p.pos[:2]))
    rgb, fb = percept.tick()
    cv2.imwrite(str(out / "_gateA.png"), overlay.draw(rgb, ex.goal, scene, ex.plan, fb))
    gateA = distA < 0.75 and p.upright
    print(f"GATE A: dist={distA:.2f} upright={p.upright} -> {'PASS' if gateA else 'FAIL'}")

    # Gate B
    hid = c.model.body("right_rubber_hand").id
    z0 = float(c.data.xpos[hid][2])
    execute("raise your right hand", 3.0)
    p = c.get_proprio(); zf = float(c.data.xpos[hid][2])
    rgb, fb = percept.tick()
    cv2.imwrite(str(out / "_gateB.png"), overlay.draw(rgb, ex.goal, scene, ex.plan, fb))
    gateB = (zf - z0) > 0.2 and p.upright
    print(f"GATE B: hand z {z0:.2f}->{zf:.2f} upright={p.upright} -> {'PASS' if gateB else 'FAIL'}")

    # Gate C — robot can actually see (real ER vision caption)
    import mujoco
    mujoco.mj_resetDataKeyframe(c.model, c.data, 0); mujoco.mj_forward(c.model, c.data)
    execute("what do you see?", 0.5)
    cap = ex.scene.caption
    gateC = bool(ex.vision and ex.vision._client) and "unavailable" not in cap and len(cap) > 10
    rgb, fb = percept.tick()
    cv2.imwrite(str(out / "_gateC.png"), overlay.draw(rgb, ex.goal, scene, ex.plan, fb))
    print(f"GATE C: caption={cap!r} -> {'PASS' if gateC else 'FAIL'}")

    # Gate D — see the red circle and walk to it (visual servoing, no hardcoded coord)
    mujoco.mj_resetDataKeyframe(c.model, c.data, 0); mujoco.mj_forward(c.model, c.data)
    execute("find the red circle and go there", 16.0)
    p = c.get_proprio()
    distD = float(np.linalg.norm(np.array([2.5, 0.0]) - p.pos[:2]))
    rgb, fb = percept.tick()
    cv2.imwrite(str(out / "_gateD.png"), overlay.draw(rgb, ex.goal, scene, ex.plan, fb))
    gateD = distD < 1.0 and p.upright
    print(f"GATE D: dist_to_disk={distD:.2f} upright={p.upright} arrived={ex._arrived_v} -> {'PASS' if gateD else 'FAIL'}")

    percept.close()
    gates = {"A": gateA, "B": gateB, "C": gateC, "D": gateD}
    print(f"\nRESULT: {gates}  (frames: _gateA/B/C/D.png)")
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", action="store_true", help="headless 2-gate acceptance test")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="optional safety time cap in seconds; 0 = run until you quit")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    a = ap.parse_args()
    raise SystemExit(run_script() if a.script else run_interactive(a.seconds, a.width, a.height))
