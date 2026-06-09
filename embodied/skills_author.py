"""3.5 Flash skill sub-agent (M3.6): author a NEW motion skill when none exists.

When the ER orchestrator meets a task with no matching skill, it delegates here. Gemini 3.5
Flash writes a structured *skill spec* (which named upper-body DOFs to move, to what target,
over what ramp, with an optional bounded oscillation). `synthesize.synthesize_from_spec` turns
that spec into a GMT reference motion the controller can track. This is pure data — no code or
web download, and every DOF target is clipped to a safe range — so an "invented" skill can move
the arms/waist but cannot drive unstable leg motions.
"""
from __future__ import annotations

import json
import re

import synthesize

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# Flash models tried in order; if one is rate-limited (429) we fall through to the next, so
# skill authoring keeps working even when the newest model's free quota is exhausted.
FLASH_MODELS = ["gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.0-flash"]

SPEC_PROMPT = """You design a short upper-body motion for a humanoid robot (Unitree G1) by
emitting ONE JSON "skill spec". You may ONLY move these named DOFs (radians):
  R_SH_PITCH/L_SH_PITCH  (shoulder pitch; more negative = arm raised forward/up)
  R_SH_ROLL/L_SH_ROLL    (shoulder roll; R negative = out to the right side, L positive = left)
  R_SH_YAW/L_SH_YAW, R_ELBOW/L_ELBOW (elbow positive = bend)
  WAIST_YAW, WAIST_ROLL, WAIST_PITCH (pitch positive = lean/bow forward)
The legs are NOT available (balance is handled separately) — design arm/waist gestures only.
Return ONLY JSON:
{"name":"<short_snake_name>","seconds":3.0,
 "channels":[{"dof":"R_SH_PITCH","target":-1.6,"ramp":0.3}],
 "oscillate":{"dof":"R_ELBOW","hz":1.5,"amp":0.4}}
Use "oscillate" only for repetitive motions (wave, clap-like); otherwise omit it. Pick targets
that visibly express the requested action. Keep it physically gentle."""


def author_skill(name: str, description: str, client, models=None, types=None) -> dict | None:
    """Ask Flash for a skill spec for `description`; returns a validated spec dict or None.

    Tries each model in `models` (default FLASH_MODELS), skipping ones that are rate-limited,
    so authoring survives an exhausted free-tier quota on the newest model. `client`/`types` are
    a google-genai client + types module (as built in vision.py/planner.py). Returns None if no
    model produced a usable spec; the caller then maps to the closest existing skill.
    """
    if client is None:
        return None
    for model in (models or FLASH_MODELS):
        try:
            cfg = types.GenerateContentConfig(temperature=0.2) if types else None
            resp = client.models.generate_content(
                model=model,
                contents=[SPEC_PROMPT, f"Action requested: {name} — {description}"],
                config=cfg,
            )
            text = getattr(resp, "text", "") or ""
            m = JSON_RE.search(text)
            if not m:
                continue
            spec = _sanitize(json.loads(m.group(0)), fallback_name=name)
            if spec is not None:
                return spec
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                continue   # try the next flash model
            continue
    return None


def _sanitize(spec: dict, fallback_name: str) -> dict | None:
    if not isinstance(spec, dict):
        return None
    name = str(spec.get("name") or fallback_name).strip().replace(" ", "_")[:40]
    chans = []
    for ch in spec.get("channels", []) or []:
        dof = str(ch.get("dof", "")).upper()
        if dof in synthesize.DOF and isinstance(ch.get("target", None), (int, float)):
            chans.append({"dof": dof, "target": float(ch["target"]),
                          "ramp": float(ch.get("ramp", 0.3))})
    if not chans:
        return None   # nothing usable -> let caller fall back to closest existing skill
    out = {"name": name or fallback_name, "seconds": float(spec.get("seconds", 3.0)),
           "channels": chans}
    osc = spec.get("oscillate")
    if isinstance(osc, dict) and str(osc.get("dof", "")).upper() in synthesize.DOF:
        out["oscillate"] = {"dof": str(osc["dof"]).upper(),
                            "hz": float(osc.get("hz", 1.5)), "amp": float(osc.get("amp", 0.3))}
    return out


def build_skill(name: str, description: str, client, models=None, types=None):
    """Author + synthesize in one step. Returns (skill_name, motion_path) or (None, None)."""
    spec = author_skill(name, description, client, models=models, types=types)
    if spec is None:
        return None, None
    path = synthesize.synthesize_from_spec(spec)
    return spec["name"], path
