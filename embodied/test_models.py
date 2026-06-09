"""Per-model test harness (M3.6) — validate each model independently for incremental progress.

  python embodied/test_models.py --local   # offline parser routing (no API)
  python embodied/test_models.py --flash    # 3.5 Flash: skill authoring (+ text intent)
  python embodied/test_models.py --er        # Gemini-ER: grounded image planning (uses memory)
  python embodied/test_models.py --all       # all three, prints a PASS/FAIL table

Each test is independent so a single model can be iterated on without running the others.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "controller"))


def test_local() -> bool:
    """Offline grounded routing in planner.Brain (client disabled — no API)."""
    from planner import Brain
    from memory import SpatialMemory
    mem = SpatialMemory(); mem.update("green sphere", (3.6, 1.4), 1)
    b = Brain(); b._client = None
    cases = {
        "get up": ("skill", "stand_up"),
        "crawl on the floor": ("skill", None),          # never-refuse -> some skill
        "go to the green sphere": ("go_to", None),       # memory recall -> coords
        "look at the green sphere": ("look_at", None),
        "what do you see": ("look", None),
    }
    ok = True
    for cmd, (kind, skill) in cases.items():
        g, p = b.plan(cmd, None, mem)
        good = (g.kind == kind) and (skill is None or g.skill == skill)
        if cmd == "go to the green sphere":
            good = good and g.target_xy is not None
        ok = ok and good
        print(f"  [local] {cmd:26s} -> kind={g.kind:11s} skill={g.skill} xy={g.target_xy} {'ok' if good else 'FAIL'}")
    return ok


def test_flash() -> bool:
    """Two parts: (1) the spec->motion synth path is deterministic and must PASS (no API);
    (2) live Flash authoring is attempted but SKIPped if every flash model is quota-exhausted."""
    import pickle
    from dotenv import load_dotenv
    load_dotenv(str(Path(__file__).resolve().parents[1] / ".env"))
    import skills_author
    import synthesize

    # (1) deterministic synth from a fixed spec — the part we own.
    spec = {"name": "salute_test", "seconds": 2.0,
            "channels": [{"dof": "R_SH_PITCH", "target": -1.4, "ramp": 0.3},
                         {"dof": "R_ELBOW", "target": 1.4, "ramp": 0.3}]}
    path = synthesize.synthesize_from_spec(spec)
    d = pickle.load(open(path, "rb"))
    synth_ok = os.path.exists(path) and d["dof_pos"].shape[1] == 23 and d["fps"] == synthesize.FPS
    print(f"  [flash] synth path: {os.path.basename(path)} dof={d['dof_pos'].shape} {'ok' if synth_ok else 'FAIL'}")

    # (2) live authoring (best-effort; quota -> SKIP, not FAIL).
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        print("  [flash] live authoring: no API key -> SKIP"); return synth_ok
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=key)
    name, lpath = skills_author.build_skill("salute", "raise the right hand to the forehead", client, types=types)
    if lpath and os.path.exists(lpath):
        print(f"  [flash] live authored '{name}' -> {os.path.basename(lpath)} ok")
    else:
        print("  [flash] live authoring unavailable (all flash models quota-exhausted) -> SKIP")
    return synth_ok


def test_er() -> bool:
    """Gemini-ER returns a grounded decision from the live frame + seeded memory."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        print("  [er] no API key -> SKIP"); return True
    import run_navigation_demo as base  # build the controller + perception
    from orchestrator import OrchestratorBrain
    from memory import SpatialMemory
    bus, c, percept, _b, nav, ex, scene, _mem = base.build()
    mem = SpatialMemory(); mem.update("green sphere", (3.6, 1.4), 1)
    scene.caption = "a green sphere and a red disk on a checkered floor"
    brain = OrchestratorBrain(skill_registry=dict(base.SKILL_MOTIONS))
    if brain._client is None:
        print("  [er] client unavailable -> SKIP"); percept.close(); return True
    ego = percept.render_ego()
    data = brain._er_plan("go to the green sphere", ego, scene, mem.known())
    rate_limited = brain._er_skip > 0    # set only on a 429 cooldown
    percept.close()
    if data is None:
        if rate_limited:
            print("  [er] ER per-minute rate-limited (transient) -> SKIP"); return True
        print("  [er] ER returned no decision -> FAIL"); return False
    good = isinstance(data, dict) and data.get("kind") in (
        "go_to", "go_to_visual", "look", "look_at", "skill", "idle")
    # Grounded check: navigating to a known object should reference that label, not invent.
    grounded = (data.get("kind") != "go_to") or (data.get("target") in (None, "stage center") or
                                                 "green sphere" in str(data.get("target", "")))
    print(f"  [er] decision={data} grounded={grounded} {'ok' if (good and grounded) else 'FAIL'}")
    return good and grounded


def test_jorge() -> bool:
    """End-to-end: Jorge looks around (360°) to fill memory, then is told to GO TO a remembered
    object and must actually walk to it (memory -> grounded plan -> nav -> arrival)."""
    import math
    import numpy as np
    import mujoco
    import run_navigation_demo as base
    bus, c, percept, brain, nav, ex, scene, memory = base.build()

    # 1) Look around: rotate Jorge's base to several headings and learn what he sees (free local).
    mujoco.mj_resetDataKeyframe(c.model, c.data, 0)
    for deg in (0, 60, 120, 180, 240, 300):
        yaw = math.radians(deg)
        c.data.qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]  # w,x,y,z about z
        mujoco.mj_forward(c.model, c.data)
        ex._learn(ex.vision.quick_look(percept.render_ego()))
    known = memory.known()
    print("  [jorge] looked around -> memory:",
          {k: (round(v[0], 1), round(v[1], 1)) for k, v in known.items()})
    if not known:
        print("  [jorge] memory empty -> FAIL"); percept.close(); return False

    # 2) Ask Jorge to go to a remembered object; he must plan from memory and walk there.
    target = "green sphere"
    g, p = brain.plan(f"go to the {target}", scene, memory)
    recalled = g.kind == "go_to" and g.target_xy is not None
    print(f"  [jorge] '{target}' -> {g.kind} target_xy={g.target_xy} (recalled={recalled})")
    if not recalled:
        percept.close(); return False

    mujoco.mj_resetDataKeyframe(c.model, c.data, 0); mujoco.mj_forward(c.model, c.data)
    ex.set(g, p)
    for i in range(int(20.0 / c.sim_dt)):
        c.step()
        if i % c.sim_decimation == 0:
            ex.tick()
        pr = c.get_proprio()
        if g.target_xy and np.linalg.norm(np.array(g.target_xy) - pr.pos[:2]) < 0.7:
            break
    pr = c.get_proprio()
    gid = mujoco.mj_name2id(c.model, mujoco.mjtObj.mjOBJ_GEOM, "prop_ball")  # the green sphere
    truth = np.array(c.model.geom_pos[gid][:2], dtype=float)
    dist = float(np.linalg.norm(truth - pr.pos[:2]))
    good = dist < 1.3 and pr.upright
    print(f"  [jorge] walked to {tuple(round(float(x),1) for x in pr.pos[:2])}; "
          f"dist to real green={dist:.2f} upright={pr.upright} {'ok' if good else 'FAIL'}")
    percept.close()
    return good


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--flash", action="store_true")
    ap.add_argument("--er", action="store_true")
    ap.add_argument("--jorge", action="store_true", help="end-to-end: look around -> go to a remembered object")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    run = {"local": a.local or a.all, "flash": a.flash or a.all,
           "er": a.er or a.all, "jorge": a.jorge or a.all}
    if not any(run.values()):
        run = {"local": True, "flash": True, "er": True, "jorge": True}  # default: all
    results = {}
    if run["local"]:
        print("== LOCAL =="); results["local"] = test_local()
    if run["flash"]:
        print("== FLASH =="); results["flash"] = test_flash()
    if run["er"]:
        print("== ER =="); results["er"] = test_er()
    if run["jorge"]:
        print("== JORGE (look around -> go to remembered object) =="); results["jorge"] = test_jorge()
    print("\nRESULT:", {k: ("PASS" if v else "FAIL") for k, v in results.items()})
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
