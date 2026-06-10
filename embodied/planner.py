"""Planner / orchestrator: natural-language command -> Goal + Plan.

GROUNDED + never-refuse. The planner reasons over what the robot ACTUALLY knows — the real
skill list, the spatial memory (objects seen + their world coords), and the latest vision
caption — and never invents objects, positions, or skills:
  * Local-first: a robust, typo/paraphrase-tolerant keyword parser handles commands instantly
    and offline (so the demo works even when the Gemini free-tier quota is exhausted).
  * Gemini 3.5 Flash is the backup for phrasings the local parser can't resolve; it is fed a
    context block (available skills, known objects+coords, current caption) and forbidden from
    fabricating coordinates.
  * Never refuse: unknown commands are difflib-matched to the closest real skill rather than
    returning "not supported". `idle` only for empty/greeting input.

Swappable Brain interface so a different model could drop in later.
"""
from __future__ import annotations

import difflib
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from messaging import Goal, Plan, SceneView

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# Landmarks (walk targets) — the red stage disk at (2.5, 0); home = the spawn point.
STAGE = (2.5, 0.0)
HOME = (0.0, 0.0)
LANDMARK_WORDS = ["stage", "center", "centre", "disk", "disc", "circle", "podium", "mark", "spot"]

# All available skills (synthesized motions + steering turns + recover-to-stand + signals).
ALL_SKILLS = ["raise_right_hand", "raise_left_hand", "raise_both_hands", "wave_right",
              "turn_left", "turn_right", "bow", "nod", "clap", "celebrate",
              "point_right", "point_left", "halt_sign", "wave_in", "stand", "stand_up"]

SYSTEM = """You translate a command for a humanoid robot into ONE JSON object.
You are GROUNDED: you may ONLY use the skills and known objects listed in the context.
Rules:
- Pick the single CLOSEST listed skill for any gesture/motion command. NEVER refuse; never
  say "not supported". Use "idle" ONLY for an empty input or a pure greeting.
- For navigation: target a known object label (from the context) or the "stage center"
  landmark. NEVER invent coordinates.
- If the user names an object you have NOT seen yet (not in known objects), emit
  "go_to_visual" (go find it) or "look_at" — do not guess where it is.
Return ONLY JSON:
{"kind":"go_to|go_to_visual|look|look_at|skill|idle",
 "target":"stage center"|"<known object label>"|null,
 "skill":<one listed skill or null>,
 "reasoning":"short"}
"""


def _fuzzy_in(tokens: list[str], *keys: str, cutoff: float = 0.84) -> bool:
    """True if any key matches a token: exact, shared prefix (>=4 chars), or close ratio.

    Strict on short tokens to avoid false positives (e.g. 'in' must NOT match 'point').
    """
    for key in keys:
        for tok in tokens:
            if tok == key:
                return True
            if len(tok) >= 4 and len(key) >= 4:
                if tok.startswith(key) or key.startswith(tok):
                    return True
                if difflib.SequenceMatcher(None, tok, key).ratio() >= cutoff:
                    return True
    return False


class Brain:
    def __init__(self, model: str = "gemini-3.5-flash") -> None:
        self.model = model
        self._client = None
        self._quota_note = False
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if key:
            try:
                from google import genai
                from google.genai import types
                self._client = genai.Client(api_key=key)
                self._types = types
            except Exception:
                self._client = None

    def plan(self, command: str, scene: SceneView | None = None,
             memory=None) -> tuple[Goal, Plan]:
        """Grounded planning: local parser first (instant/offline), then Gemini backup fed the
        real skills + remembered objects + current caption, then a difflib never-refuse guess."""
        known = memory.known() if memory is not None else {}
        data = self._local(command, known)
        if data["kind"] == "idle" and self._client:
            llm = self._llm(command, scene, known)   # grounded backup
            if llm is not None:
                data = llm
        if data["kind"] == "idle":
            data = self._guess(command, known)        # never-refuse last resort
        return self._to_goal(command, data, known)

    # --- robust local parser ---------------------------------------------
    def _local(self, command: str, known: dict | None = None) -> dict:
        known = known or {}
        c = command.lower()
        toks = re.findall(r"[a-z0-9]+", c)
        raise_word = _fuzzy_in(toks, "raise", "rise", "lift", "put", "up")
        hand_word = _fuzzy_in(toks, "hand", "hands", "arm", "arms")

        # vision: "what do you see", "look", "describe"
        see_word = _fuzzy_in(toks, "see", "look", "describe", "vision", "watching", "observe")
        circle_word = _fuzzy_in(toks, "circle", "disk", "disc", "red")
        find_word = _fuzzy_in(toks, "find", "search", "locate", "spot")
        go_verb0 = _fuzzy_in(toks, "go", "walk", "move", "head", "navigate", "approach", "reach", "come")

        # SAFETY FIRST — emergency stop: halt everything (skills, nav, steering) immediately.
        if _fuzzy_in(toks, "emergency", "estop") or ("emergency" in c) or \
           (_fuzzy_in(toks, "stop") and _fuzzy_in(toks, "now", "all", "everything")):
            return {"kind": "estop", "target": None, "skill": None,
                    "reasoning": "EMERGENCY STOP - halt all motion"}

        # pick & carry (before the raise/hand branch: 'pick UP' must not read as raise-arm)
        if _fuzzy_in(toks, "pick", "grab", "fetch") or \
           (_fuzzy_in(toks, "lift", "carry") and _fuzzy_in(toks, "box", "cube", "object", "payload", "it")):
            box = "magenta box" if _fuzzy_in(toks, "magenta", "pink") else \
                  ("cyan box" if _fuzzy_in(toks, "cyan") else "box")  # bare -> nearest box
            return {"kind": "pick_up", "target": box, "skill": None,
                    "reasoning": f"pick up the {box}"}
        if (_fuzzy_in(toks, "put", "set", "drop", "place", "release") and
                _fuzzy_in(toks, "down", "it", "box", "here", "floor")):
            return {"kind": "put_down", "target": None, "skill": None,
                    "reasoning": "set the carried box down"}

        # inventory scan: 360 sweep + structured report of everything seen
        if _fuzzy_in(toks, "scan", "inventory", "stocktake") or \
           ("take stock" in c) or (_fuzzy_in(toks, "report") and _fuzzy_in(toks, "area", "objects", "stock", "what")):
            return {"kind": "scan", "target": None, "skill": None,
                    "reasoning": "scan the area and report the inventory"}

        # return home / to base / dock -> walk back to the spawn point
        if _fuzzy_in(toks, "home", "base", "dock", "charging"):
            return {"kind": "go_to", "target": "home", "skill": None,
                    "reasoning": "return to the home position"}

        # industrial hand signals
        if _fuzzy_in(toks, "halt") and not _fuzzy_in(toks, "emergency"):
            return _skill("halt_sign", "show the HALT hand signal")
        if _fuzzy_in(toks, "wave") and _fuzzy_in(toks, "in", "them", "here", "come"):
            return _skill("wave_in", "wave them in (come-here signal)")

        # "point at <remembered object>" -> face it and point
        if _fuzzy_in(toks, "point") and _fuzzy_in(toks, "at", "to", "toward", "towards"):
            obj = self._match_known(command, known)
            if obj:
                return {"kind": "point_at", "target": obj, "skill": None,
                        "reasoning": f"turn to face the {obj} and point at it"}
        if _fuzzy_in(toks, "point") and _fuzzy_in(toks, "left"):
            return _skill("point_left", "point left")

        # stand up / get up / recover -> the recover-to-stand skill (before bare 'stand')
        if _fuzzy_in(toks, "getup", "recover") or \
           (_fuzzy_in(toks, "up") and _fuzzy_in(toks, "get", "stand", "rise", "stand")) or \
           (_fuzzy_in(toks, "rise") and not hand_word):
            return _skill("stand_up", "stand up / recover to a stable standing pose")

        # "look at <remembered object>" -> recall and face it
        if see_word and _fuzzy_in(toks, "at", "back", "toward", "towards"):
            obj = self._match_known(command, known)
            if obj:
                return {"kind": "look_at", "target": obj, "skill": None,
                        "reasoning": f"turn back to look at the {obj}"}

        # Explicit visual SEARCH ("find/search/locate the red circle and go there") — Gate D:
        # spin to acquire the disk by sight, then servo to it. Only when the user asks to *find*.
        if circle_word and find_word:
            return {"kind": "go_to_visual", "target": "red circle", "skill": None,
                    "reasoning": "find the red circle visually and walk to it"}
        # "go to the red circle": the disk is the stage landmark at a KNOWN position (2.5,0) —
        # walk straight there reliably instead of a blind visual spin.
        if circle_word and (go_verb0 or _fuzzy_in(toks, "there")):
            return {"kind": "go_to", "target": "stage center", "skill": None,
                    "reasoning": "walk to the red circle (the stage disk at 2.5,0)"}
        if see_word and not go_verb0:
            return {"kind": "look", "target": None, "skill": None, "reasoning": "describe what I see"}

        # gestures
        if _fuzzy_in(toks, "wave"):
            return _skill("wave_right", "wave")
        if _fuzzy_in(toks, "clap", "applaud"):
            return _skill("clap", "clap hands")
        if _fuzzy_in(toks, "bow"):
            return _skill("bow", "bow")
        if _fuzzy_in(toks, "nod"):
            return _skill("nod", "nod")
        if _fuzzy_in(toks, "celebrate", "cheer", "hooray", "yay"):
            return _skill("celebrate", "celebrate")
        if _fuzzy_in(toks, "point"):
            return _skill("point_right", "point")
        if _fuzzy_in(toks, "turn", "rotate", "spin"):
            if _fuzzy_in(toks, "right"):
                return _skill("turn_right", "turn right")
            return _skill("turn_left", "turn left")
        if raise_word or hand_word or _fuzzy_in(toks, "air"):
            both = _fuzzy_in(toks, "both", "two", "2", "all") or (hand_word and not _fuzzy_in(toks, "right", "left"))
            if _fuzzy_in(toks, "right") and not _fuzzy_in(toks, "both", "two"):
                return _skill("raise_right_hand", "raise right hand")
            if _fuzzy_in(toks, "left") and not _fuzzy_in(toks, "both", "two"):
                return _skill("raise_left_hand", "raise left hand")
            if both or hand_word or _fuzzy_in(toks, "air"):
                return _skill("raise_both_hands", "raise both hands")
        if _fuzzy_in(toks, "stop", "stand", "halt", "stay", "still", "idle", "wait", "freeze"):
            return _skill("stand", "stand still")

        # navigation
        go_verb = _fuzzy_in(toks, "go", "walk", "move", "head", "navigate", "approach", "reach", "come")
        # named object the robot has already seen -> walk to its remembered position
        if go_verb:
            obj = self._match_known(command, known)
            if obj and not _fuzzy_in(toks, *LANDMARK_WORDS):
                return {"kind": "go_to", "target": obj, "skill": None,
                        "reasoning": f"walk to the remembered {obj}"}
        # landmark (stage / red disk)
        if _fuzzy_in(toks, *LANDMARK_WORDS) or (go_verb and _fuzzy_in(toks, "there", "forward", "ahead", "red")):
            return {"kind": "go_to", "target": "stage center", "skill": None,
                    "reasoning": "walk to the stage (red disk)"}
        if go_verb:
            # a go-verb with an unknown object name -> go find it visually, don't invent coords
            obj = _trailing_noun(command)
            if obj:
                return {"kind": "go_to_visual", "target": obj, "skill": None,
                        "reasoning": f"haven't seen a {obj} yet — go look for it"}
            return {"kind": "go_to", "target": "stage center", "skill": None,
                    "reasoning": "walk forward to the stage"}
        return {"kind": "idle", "target": None, "skill": None, "reasoning": "no local match"}

    def _match_known(self, command: str, known: dict) -> str | None:
        """Return the remembered object label best matching the command, if any."""
        if not known:
            return None
        from memory import _tokens  # reuse the same tokenizer
        ct = _tokens(command)
        best, score = None, 0
        for lab in known:
            s = len(ct & _tokens(lab))
            if s > score:
                best, score = lab, s
        if best and score > 0:
            return best
        m = difflib.get_close_matches(command.lower(), list(known), n=1, cutoff=0.6)
        return m[0] if m else None

    # --- LLM backup (grounded) -------------------------------------------
    def _context(self, scene: SceneView | None, known: dict) -> str:
        objs = ", ".join(f"{k} -> ({v[0]:.1f},{v[1]:.1f})" for k, v in known.items()) or "(none yet)"
        cap = (scene.caption if scene else "") or "(nothing observed yet)"
        return (f"Available skills: {ALL_SKILLS}\n"
                f"Known objects (label -> world x,y): {objs}\n"
                f"Currently seeing: \"{cap}\"")

    def _llm(self, command: str, scene: SceneView | None, known: dict) -> dict | None:
        try:
            resp = self._client.models.generate_content(
                model=self.model,
                contents=[SYSTEM, self._context(scene, known), f"Command: {command}"],
                config=self._types.GenerateContentConfig(temperature=0.0),
            )
            text = getattr(resp, "text", "") or ""
            m = JSON_RE.search(text)
            return json.loads(m.group(0)) if m else None
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                if not self._quota_note:
                    print("  [note] Gemini free-tier quota exhausted — using local parser only.")
                    self._quota_note = True
            return None

    # --- never-refuse offline guess --------------------------------------
    def _guess(self, command: str, known: dict) -> dict:
        """Last resort: difflib-match the command to the closest real skill (or remembered
        object) so the robot always does *something* grounded, never refuses."""
        # a remembered object mentioned -> go there
        obj = self._match_known(command, known)
        if obj:
            return {"kind": "go_to", "target": obj, "skill": None,
                    "reasoning": f"closest grounded match: go to the {obj}"}
        skill = self._closest_skill(command)
        if skill:
            return _skill(skill, f"closest skill guess for '{command.strip()}'")
        return {"kind": "idle", "target": None, "skill": None, "reasoning": "nothing to do"}

    def _closest_skill(self, name: str) -> str | None:
        """Closest listed skill to free-text, comparing whole phrase and per-token."""
        toks = re.findall(r"[a-z0-9]+", name.lower())
        if not toks:
            return None
        cands = [name.lower().replace(" ", "_")] + toks
        for cand in cands:
            m = difflib.get_close_matches(cand, ALL_SKILLS, n=1, cutoff=0.55)
            if m:
                return m[0]
        # token overlap against skill names (e.g. "crawl down low" -> bow via 'down'? fall to stand)
        best, score = None, 0.0
        for sk in ALL_SKILLS:
            sktoks = set(sk.split("_"))
            r = max((difflib.SequenceMatcher(None, t, s).ratio() for t in toks for s in sktoks),
                    default=0.0)
            if r > score:
                best, score = sk, r
        return best if score >= 0.5 else "stand"  # default to a safe stable pose, never idle

    def _to_goal(self, command: str, data: dict, known: dict) -> tuple[Goal, Plan]:
        kind = data.get("kind", "idle")
        reasoning = data.get("reasoning", "")
        target = data.get("target")
        if kind == "estop":
            goal = Goal(kind="estop", text=command)
            plan = Plan(steps=["halt all motion"], reasoning=reasoning, current="EMERGENCY STOP")
        elif kind == "scan":
            goal = Goal(kind="scan", text=command)
            plan = Plan(steps=["rotate 360", "log every object + position", "report inventory"],
                        reasoning=reasoning, current="scanning the area")
        elif kind == "pick_up":
            goal = Goal(kind="pick_up", target_name=target or "box", text=command)
            plan = Plan(steps=["walk to the box", "bend and lift", "carry"],
                        reasoning=reasoning, current="going to pick up the box")
        elif kind == "put_down":
            goal = Goal(kind="put_down", text=command)
            plan = Plan(steps=["bend down", "release", "stand"],
                        reasoning=reasoning, current="setting the box down")
        elif kind == "point_at":
            goal = Goal(kind="point_at", target_name=target, text=command)
            plan = Plan(steps=[f"face the {target}", "point at it"],
                        reasoning=reasoning, current=f"pointing at {target}")
        elif kind == "look":
            goal = Goal(kind="look", text=command)
            plan = Plan(steps=["look through onboard camera", "describe scene"],
                        reasoning=reasoning or "describe what I see", current="looking")
        elif kind == "look_at":
            goal = Goal(kind="look_at", target_name=target, text=command)
            plan = Plan(steps=[f"recall {target}", "turn to face it", "describe"],
                        reasoning=reasoning, current=f"looking back at {target}")
        elif kind == "go_to_visual":
            goal = Goal(kind="go_to_visual", target_name=target, text=command)
            plan = Plan(steps=[f"find {target or 'target'}", "steer toward it", "walk", "stop when close"],
                        reasoning=reasoning, current=f"searching for {target or 'the target'}")
        elif kind == "go_to":
            # landmark -> STAGE/HOME coords; named known object -> recall its remembered position.
            if target in ("home", "base", "dock"):
                goal = Goal(kind="go_to", target_xy=HOME, target_name="home", text=command)
                plan = Plan(steps=["walk back to the home position", "stop on arrival"],
                            reasoning=reasoning, current="returning home")
            elif target and target not in ("stage center", "stage", "center", "centre") and target in known:
                goal = Goal(kind="go_to", target_xy=known[target], target_name=target, text=command)
                plan = Plan(steps=[f"recall {target}", "walk there", "stop on arrival"],
                            reasoning=reasoning, current=f"navigating to {target}")
            elif target and target in known:
                goal = Goal(kind="go_to", target_xy=known[target], target_name=target, text=command)
                plan = Plan(steps=["walk there", "stop on arrival"], reasoning=reasoning,
                            current=f"navigating to {target}")
            elif target and target not in ("stage center", "stage", "center", "centre"):
                # named but not yet seen -> go find it visually rather than inventing coords
                goal = Goal(kind="go_to_visual", target_name=target, text=command)
                plan = Plan(steps=[f"haven't seen {target}", "look for it", "walk to it"],
                            reasoning=reasoning or f"go find the {target}",
                            current=f"searching for {target}")
            else:
                goal = Goal(kind="go_to", target_xy=STAGE, text=command)
                plan = Plan(steps=["face stage", "walk forward", "stop on arrival"],
                            reasoning=reasoning, current="navigating to stage")
        elif kind == "skill":
            skill = data.get("skill")
            if skill not in ALL_SKILLS:
                skill = self._closest_skill(skill or command) or "stand"
            goal = Goal(kind="skill", skill=skill, text=command)
            plan = Plan(steps=[f"perform {skill}"], reasoning=reasoning, current=skill)
        else:
            goal = Goal(kind="idle", text=command)
            plan = Plan(steps=["idle"], reasoning=reasoning or "command not understood", current="idle")
        return goal, plan


def _skill(name: str, reason: str) -> dict:
    return {"kind": "skill", "target": None, "skill": name, "reasoning": reason}


_STOPWORDS = {"go", "walk", "move", "head", "navigate", "approach", "reach", "come", "to",
              "the", "a", "an", "towards", "toward", "at", "over", "there", "please", "and",
              "then", "near", "by", "into", "on"}


def _trailing_noun(command: str) -> str | None:
    """Extract the object phrase from a go-command, e.g. 'go to the green box' -> 'green box'."""
    toks = [t for t in re.findall(r"[a-z0-9]+", command.lower()) if t not in _STOPWORDS]
    phrase = " ".join(toks).strip()
    return phrase or None
