"""Orchestrator brain (M3.6): Gemini-ER is the BOSS, Gemini 3.5 Flash is a skill sub-agent.

ER sees the onboard frame + the grounded context (available skills, known objects from spatial
memory with world coords, current caption) and decides ONE action for every command. When the
task needs a skill the robot doesn't have, ER delegates to the 3.5 Flash sub-agent
(`skills_author`) which AUTHORS a new motion; it is synthesized, cached, and registered at
runtime. If ER is momentarily rate-limited or errors, we fall back to the classic `planner.Brain`
(local parser -> 3.5 text -> closest-skill guess) so the demo never freezes.

This is additive: the classic `planner.Brain` / `run_navigation_demo.py` are untouched.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import cv2
from dotenv import load_dotenv

import planner
import skills_author
from messaging import SceneView

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

ER_MODEL = "gemini-robotics-er-1.6-preview"
FLASH_MODEL = "gemini-3.5-flash"

ER_SYSTEM = """You are the decision-making brain of a humanoid robot (Unitree G1). You SEE the
onboard camera image and you are given the robot's known skills, the objects it remembers (with
world x,y), and a caption. Decide ONE action for the user's command. Be GROUNDED:
- Navigate only to a KNOWN object label or the "stage center" landmark — NEVER invent coordinates.
  If the target hasn't been seen, use "go_to_visual" (go find it).
- For a gesture/motion, pick the CLOSEST existing skill. If NONE fits the request, propose a new
  skill via "new_skill" (a short name + a one-line description of the arm/waist motion). Never refuse.
- Industry tasks: "scan" = 360 inventory sweep + report; "pick_up" = fetch the cyan carry box;
  "put_down" = set the carried box down; "point_at" = face a known object and point;
  "estop" = EMERGENCY STOP, halt everything (use for any urgent stop request).
- "idle" only for empty input or a pure greeting.
Return ONLY JSON:
{"kind":"go_to|go_to_visual|look|look_at|skill|scan|pick_up|put_down|point_at|estop|idle",
 "target":"<known label or stage center>"|null,
 "skill":"<one existing skill>"|null,
 "new_skill":{"name":"<snake>","description":"<what the arms/waist do>"}|null,
 "reasoning":"short"}
"""


class OrchestratorBrain:
    def __init__(self, skill_registry: dict | None = None, er_period_s: float = 30.0) -> None:
        self.fallback = planner.Brain()          # classic local -> 3.5 text -> guess
        self.skill_registry = skill_registry if skill_registry is not None else {}
        self.er_period_s = er_period_s           # call ER (image) at most once per this many seconds
        self._last_er = -1e9                      # monotonic time of the last ER call
        self._client = None
        self._types = None
        self._er_skip = 0                        # skip ER for N commands after a rate-limit
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if key:
            try:
                from google import genai
                from google.genai import types
                self._client = genai.Client(api_key=key)
                self._types = types
            except Exception:
                self._client = None

    # ---- public API (mirrors planner.Brain.plan, plus the live image) ----
    def plan(self, command: str, scene: SceneView | None = None, memory=None, image=None):
        known = memory.known() if memory is not None else {}
        # Money-saver: only spend an ER (image) call at most once per `er_period_s`. Between ER
        # turns the cheap local/3.5 planner handles commands, grounded on memory + the last ER
        # scene/caption. (er_period_s=0 -> ER on every command, the old behavior.)
        now = time.monotonic()
        due = (now - self._last_er) >= self.er_period_s
        if image is not None and self._client is not None and self._er_skip <= 0 and due:
            data = self._er_plan(command, image, scene, known)
            if data is not None:
                self._last_er = now
                return self._finalize(command, data, known, brain="ER")
        elif self._er_skip > 0:
            self._er_skip -= 1
        # Fallback / between-ER: classic planner (already grounded + never-refuse).
        goal, plan = self.fallback.plan(command, scene, memory)
        plan.brain = "3.5" if (self.fallback._client and "quota" not in plan.reasoning) else "local"
        # If the fallback only found a weak closest-skill guess for a novel action, AUTHOR a real
        # skill instead (cheap 3.5 text sub-agent, model-fallback — not the throttled ER image
        # call). This lets Jorge "learn" moves like 'dance' between ER turns, too.
        if goal.kind == "skill" and self._client is not None:
            r = (plan.reasoning or "").lower()
            if "closest" in r or "guess" in r:
                self._author_for(command, goal, plan)
        return goal, plan

    def _author_for(self, command: str, goal, plan) -> None:
        """Author (or reuse) a skill for a novel command and point the goal at it."""
        name = self._skill_name(command)
        if name in self.skill_registry:                 # already learned this -> reuse
            skname = name
        else:
            skname, path = skills_author.build_skill(name, command, self._client, types=self._types)
            if not path:
                return                                  # all flash models unavailable -> keep guess
            self.skill_registry[skname] = path
            if skname not in planner.ALL_SKILLS:
                planner.ALL_SKILLS.append(skname)
        goal.skill = skname
        plan.reasoning = f"authored skill '{skname}' for '{command.strip()}'"
        plan.current = skname
        plan.brain = "3.5*"                             # * = authored a new skill

    @staticmethod
    def _skill_name(command: str) -> str:
        c = command.lower().strip().rstrip("?.!")
        for pre in ("can you ", "could you ", "would you ", "please ", "try to ",
                    "i want you to ", "learn to ", "do a ", "do an ", "do the ", "do ", "go "):
            if c.startswith(pre):
                c = c[len(pre):]
        c = re.sub(r"[^a-z0-9]+", "_", c).strip("_")
        return c[:24] or "move"

    # ---- ER planning ----
    def _context(self, scene: SceneView | None, known: dict) -> str:
        objs = ", ".join(f"{k} -> ({v[0]:.1f},{v[1]:.1f})" for k, v in known.items()) or "(none yet)"
        cap = (scene.caption if scene else "") or "(nothing observed yet)"
        skills = sorted(set(planner.ALL_SKILLS) | set(self.skill_registry))
        return (f"Available skills: {skills}\n"
                f"Known objects (label -> world x,y): {objs}\n"
                f"Currently seeing: \"{cap}\"")

    def _er_plan(self, command: str, image, scene, known: dict) -> dict | None:
        try:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                return None
            part = self._types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg")
            resp = self._client.models.generate_content(
                model=ER_MODEL,
                contents=[ER_SYSTEM, self._context(scene, known), f"Command: {command}", part],
                config=self._types.GenerateContentConfig(temperature=0.0),
            )
            text = getattr(resp, "text", "") or ""
            m = JSON_RE.search(text)
            return json.loads(m.group(0)) if m else None
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                self._er_skip = 8   # cool down ER for a few commands (paid recovers)
            return None

    # ---- decision -> Goal/Plan, authoring new skills via the 3.5 sub-agent ----
    def _finalize(self, command: str, data: dict, known: dict, brain: str):
        new_skill = data.get("new_skill")
        skill = data.get("skill")
        registry_names = set(planner.ALL_SKILLS) | set(self.skill_registry)
        authored = None
        # Author when ER asked for a new skill, or named a skill we don't have.
        if data.get("kind") == "skill" and (new_skill or (skill and skill not in registry_names)):
            name = (new_skill or {}).get("name") or skill or command
            desc = (new_skill or {}).get("description") or command
            skname, path = skills_author.build_skill(name, desc, self._client,
                                                     types=self._types)
            if path:
                self.skill_registry[skname] = path
                if skname not in planner.ALL_SKILLS:
                    planner.ALL_SKILLS.append(skname)
                data["skill"] = skname
                authored = skname
        goal, plan = self.fallback._to_goal(command, data, known)
        plan.brain = brain
        if authored:
            plan.reasoning = (plan.reasoning or "") + f"  [3.5 authored skill '{authored}']"
        return goal, plan
