"""Synthesize GMT reference motions for skills (stand, gestures).

A GMT motion is a dict {fps, root_pos (N,3), root_rot (N,4 xyzw), dof_pos (N,23)} pickled
to a .pkl. The policy tracks it whole-body with balance. Standing + arm gestures are easy
to synthesize procedurally; locomotion still uses the captured walk clip (basic_walk.pkl).

This is the "skill library" reborn as motions: each skill = a generated trajectory.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

GMT = Path(__file__).resolve().parent / "controller" / "gmt"
MOTIONS = GMT / "motions"
SYNTH = MOTIONS / "_synth"
SYNTH.mkdir(exist_ok=True)

FPS = 50
STAND_H = 0.78

# 23-DoF default standing pose (matches GMTController.default_dof_pos)
DEFAULT = np.array([-0.2, 0, 0, 0.4, -0.2, 0, -0.2, 0, 0, 0.4, -0.2, 0,
                    0, 0, 0, 0, 0.4, 0, 1.2, 0, -0.4, 0, 1.2], dtype=np.float32)

# right arm dof indices: 19 sh_pitch, 20 sh_roll, 21 sh_yaw, 22 elbow
R_SH_PITCH, R_SH_ROLL, R_SH_YAW, R_ELBOW = 19, 20, 21, 22
# left arm dof indices: 15 sh_pitch, 16 sh_roll, 17 sh_yaw, 18 elbow
L_SH_PITCH, L_SH_ROLL, L_SH_YAW, L_ELBOW = 15, 16, 17, 18


def _save(name: str, root_pos, root_rot, dof_pos) -> str:
    data = {"fps": FPS,
            "root_pos": np.asarray(root_pos, dtype=np.float32),
            "root_rot": np.asarray(root_rot, dtype=np.float32),
            "dof_pos": np.asarray(dof_pos, dtype=np.float32)}
    path = SYNTH / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return str(path)


def _static(dof_seq) -> tuple:
    n = len(dof_seq)
    root_pos = np.tile([0.0, 0.0, STAND_H], (n, 1))
    root_rot = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))  # (x,y,z,w) identity
    return root_pos, root_rot, np.asarray(dof_seq, dtype=np.float32)


def make_stand(seconds: float = 2.0) -> str:
    n = int(seconds * FPS)
    dof = np.tile(DEFAULT, (n, 1))
    return _save("stand", *_static(dof))


def make_raise_right_hand(seconds: float = 3.0) -> str:
    """Ramp the right arm up overhead, hold, drawn from the standing pose."""
    n = int(seconds * FPS)
    dof = np.tile(DEFAULT, (n, 1))
    # phases: 0-30% raise, 30-100% hold
    for i in range(n):
        a = min(1.0, i / (0.3 * n))
        dof[i, R_SH_PITCH] = DEFAULT[R_SH_PITCH] + a * (-2.4 - DEFAULT[R_SH_PITCH])  # arm up
        dof[i, R_SH_ROLL] = DEFAULT[R_SH_ROLL] + a * (-0.15 - DEFAULT[R_SH_ROLL])
        dof[i, R_ELBOW] = DEFAULT[R_ELBOW] + a * (0.3 - DEFAULT[R_ELBOW])           # straighten
    return _save("raise_right_hand", *_static(dof))


def make_raise_left_hand(seconds: float = 3.0) -> str:
    n = int(seconds * FPS)
    dof = np.tile(DEFAULT, (n, 1))
    for i in range(n):
        a = min(1.0, i / (0.3 * n))
        dof[i, L_SH_PITCH] = DEFAULT[L_SH_PITCH] + a * (-2.4 - DEFAULT[L_SH_PITCH])
        dof[i, L_SH_ROLL] = DEFAULT[L_SH_ROLL] + a * (0.15 - DEFAULT[L_SH_ROLL])
        dof[i, L_ELBOW] = DEFAULT[L_ELBOW] + a * (0.3 - DEFAULT[L_ELBOW])
    return _save("raise_left_hand", *_static(dof))


def make_raise_both_hands(seconds: float = 3.0) -> str:
    n = int(seconds * FPS)
    dof = np.tile(DEFAULT, (n, 1))
    for i in range(n):
        a = min(1.0, i / (0.3 * n))
        dof[i, R_SH_PITCH] = DEFAULT[R_SH_PITCH] + a * (-2.4 - DEFAULT[R_SH_PITCH])
        dof[i, L_SH_PITCH] = DEFAULT[L_SH_PITCH] + a * (-2.4 - DEFAULT[L_SH_PITCH])
        dof[i, R_SH_ROLL] = DEFAULT[R_SH_ROLL] + a * (-0.15 - DEFAULT[R_SH_ROLL])
        dof[i, L_SH_ROLL] = DEFAULT[L_SH_ROLL] + a * (0.15 - DEFAULT[L_SH_ROLL])
        dof[i, R_ELBOW] = DEFAULT[R_ELBOW] + a * (0.3 - DEFAULT[R_ELBOW])
        dof[i, L_ELBOW] = DEFAULT[L_ELBOW] + a * (0.3 - DEFAULT[L_ELBOW])
    return _save("raise_both_hands", *_static(dof))


def make_wave_right(seconds: float = 3.0) -> str:
    """Right hand up, forearm waving side to side."""
    n = int(seconds * FPS)
    dof = np.tile(DEFAULT, (n, 1))
    for i in range(n):
        a = min(1.0, i / (0.25 * n))
        dof[i, R_SH_PITCH] = DEFAULT[R_SH_PITCH] + a * (-2.2 - DEFAULT[R_SH_PITCH])
        dof[i, R_SH_ROLL] = DEFAULT[R_SH_ROLL] + a * (-0.2 - DEFAULT[R_SH_ROLL])
        dof[i, R_ELBOW] = 0.6 + 0.5 * np.sin(2 * np.pi * 1.5 * i / FPS) * a  # wave
    return _save("wave_right", *_static(dof))


WAIST_YAW, WAIST_ROLL, WAIST_PITCH = 12, 13, 14


def make_bow(seconds: float = 3.0) -> str:
    n = int(seconds * FPS)
    dof = np.tile(DEFAULT, (n, 1))
    for i in range(n):
        ph = np.sin(np.pi * min(1.0, i / (0.8 * n)))  # bend down and back up
        dof[i, WAIST_PITCH] = DEFAULT[WAIST_PITCH] + ph * 0.5
    return _save("bow", *_static(dof))


def make_nod(seconds: float = 3.0) -> str:
    n = int(seconds * FPS)
    dof = np.tile(DEFAULT, (n, 1))
    for i in range(n):
        dof[i, WAIST_PITCH] = DEFAULT[WAIST_PITCH] + 0.25 * (0.5 - 0.5 * np.cos(2 * np.pi * 1.0 * i / FPS))
    return _save("nod", *_static(dof))


def make_clap(seconds: float = 3.0) -> str:
    n = int(seconds * FPS)
    dof = np.tile(DEFAULT, (n, 1))
    for i in range(n):
        a = min(1.0, i / (0.2 * n))
        c = 0.5 + 0.5 * np.sin(2 * np.pi * 1.5 * i / FPS)  # 0..1 clap cycle
        dof[i, R_SH_PITCH] = DEFAULT[R_SH_PITCH] + a * (-0.7 - DEFAULT[R_SH_PITCH])
        dof[i, L_SH_PITCH] = DEFAULT[L_SH_PITCH] + a * (-0.7 - DEFAULT[L_SH_PITCH])
        dof[i, R_SH_ROLL] = DEFAULT[R_SH_ROLL] + a * ((-0.55 + 0.35 * c) - DEFAULT[R_SH_ROLL])
        dof[i, L_SH_ROLL] = DEFAULT[L_SH_ROLL] + a * ((0.55 - 0.35 * c) - DEFAULT[L_SH_ROLL])
        dof[i, R_ELBOW] = 0.9
        dof[i, L_ELBOW] = 0.9
    return _save("clap", *_static(dof))


def make_celebrate(seconds: float = 3.0) -> str:
    n = int(seconds * FPS)
    dof = np.tile(DEFAULT, (n, 1))
    for i in range(n):
        a = min(1.0, i / (0.25 * n))
        pump = 0.25 * (0.5 - 0.5 * np.cos(2 * np.pi * 1.5 * i / FPS))
        dof[i, R_SH_PITCH] = DEFAULT[R_SH_PITCH] + a * (-2.4 + pump - DEFAULT[R_SH_PITCH])
        dof[i, L_SH_PITCH] = DEFAULT[L_SH_PITCH] + a * (-2.4 + pump - DEFAULT[L_SH_PITCH])
        dof[i, R_ELBOW] = DEFAULT[R_ELBOW] + a * (0.3 - DEFAULT[R_ELBOW])
        dof[i, L_ELBOW] = DEFAULT[L_ELBOW] + a * (0.3 - DEFAULT[L_ELBOW])
    return _save("celebrate", *_static(dof))


def make_point_right(seconds: float = 3.0) -> str:
    n = int(seconds * FPS)
    dof = np.tile(DEFAULT, (n, 1))
    for i in range(n):
        a = min(1.0, i / (0.3 * n))
        dof[i, R_SH_PITCH] = DEFAULT[R_SH_PITCH] + a * (-1.4 - DEFAULT[R_SH_PITCH])  # arm forward
        dof[i, R_ELBOW] = DEFAULT[R_ELBOW] + a * (0.05 - DEFAULT[R_ELBOW])           # straighten
    return _save("point_right", *_static(dof))


# --- generic, spec-driven synthesis (used by the 3.5 skill sub-agent, M3.6) -----------------
# Named DOFs a skill spec may move. Authored skills are restricted to arms + waist (upper body)
# so they stay quasi-static and upright; leg DOFs are intentionally NOT exposed.
DOF = {
    "R_SH_PITCH": R_SH_PITCH, "R_SH_ROLL": R_SH_ROLL, "R_SH_YAW": R_SH_YAW, "R_ELBOW": R_ELBOW,
    "L_SH_PITCH": L_SH_PITCH, "L_SH_ROLL": L_SH_ROLL, "L_SH_YAW": L_SH_YAW, "L_ELBOW": L_ELBOW,
    "WAIST_YAW": WAIST_YAW, "WAIST_ROLL": WAIST_ROLL, "WAIST_PITCH": WAIST_PITCH,
}
# Conservative safe ranges (radians) per named DOF — authored targets are clipped to these.
SAFE_RANGE = {
    "R_SH_PITCH": (-2.6, 0.6), "L_SH_PITCH": (-2.6, 0.6),
    "R_SH_ROLL": (-1.6, 0.2), "L_SH_ROLL": (-0.2, 1.6),
    "R_SH_YAW": (-1.2, 1.2), "L_SH_YAW": (-1.2, 1.2),
    "R_ELBOW": (-0.2, 1.6), "L_ELBOW": (-0.2, 1.6),
    "WAIST_YAW": (-0.5, 0.5), "WAIST_ROLL": (-0.4, 0.4), "WAIST_PITCH": (-0.4, 0.6),
}


def synthesize_from_spec(spec: dict) -> str:
    """Build + cache a GMT motion from a 3.5-authored skill spec. Pure data, no eval/exec.

    spec = {"name", "seconds"?, "channels":[{"dof","target","ramp"?}], "oscillate":{...}?}
    Each channel ramps a named arm/waist DOF from the standing default to a clipped target;
    optional `oscillate` adds a bounded sine on one DOF. Returns the saved .pkl path.
    """
    name = str(spec.get("name") or "authored").strip().replace(" ", "_")[:40] or "authored"
    seconds = float(spec.get("seconds", 3.0))
    seconds = max(1.0, min(6.0, seconds))
    n = int(seconds * FPS)
    dof = np.tile(DEFAULT, (n, 1))

    for ch in spec.get("channels", []) or []:
        key = str(ch.get("dof", "")).upper()
        if key not in DOF:
            continue   # ignore unknown / leg DOFs for safety
        idx = DOF[key]
        lo, hi = SAFE_RANGE[key]
        target = float(np.clip(float(ch.get("target", DEFAULT[idx])), lo, hi))
        ramp = float(np.clip(float(ch.get("ramp", 0.3)), 0.05, 1.0))
        for i in range(n):
            a = min(1.0, i / (ramp * n))
            dof[i, idx] = DEFAULT[idx] + a * (target - DEFAULT[idx])

    osc = spec.get("oscillate")
    if isinstance(osc, dict):
        key = str(osc.get("dof", "")).upper()
        if key in DOF:
            idx = DOF[key]
            lo, hi = SAFE_RANGE[key]
            hz = float(np.clip(float(osc.get("hz", 1.5)), 0.2, 3.0))
            amp = float(np.clip(float(osc.get("amp", 0.3)), 0.0, 0.6))
            for i in range(n):
                a = min(1.0, i / (0.25 * n))
                dof[i, idx] = float(np.clip(dof[i, idx] + a * amp * np.sin(2 * np.pi * hz * i / FPS),
                                            lo, hi))

    # Final safety clip across all named DOFs.
    for key, idx in DOF.items():
        lo, hi = SAFE_RANGE[key]
        dof[:, idx] = np.clip(dof[:, idx], lo, hi)
    return _save(name, *_static(dof))


def make_all() -> dict:
    return {
        "stand": make_stand(),
        "raise_right_hand": make_raise_right_hand(),
        "raise_left_hand": make_raise_left_hand(),
        "raise_both_hands": make_raise_both_hands(),
        "wave_right": make_wave_right(),
        "bow": make_bow(),
        "nod": make_nod(),
        "clap": make_clap(),
        "celebrate": make_celebrate(),
        "point_right": make_point_right(),
    }


if __name__ == "__main__":
    for k, v in make_all().items():
        print(k, "->", v)
