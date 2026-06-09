"""Spatial memory (Voyager-adapted, in-session): remember objects + world positions.

When the robot sees something (via the vision brain), it records the object's label and an
estimated world (x,y). Later commands like "go to the green sphere" or "look at the red
circle" recall that position even when the object is no longer in view — so the planner can
ground its plans in things actually seen, not invented coordinates.

In-process only: forgets on restart (no disk persistence — that is the deferred Voyager
persistence step). The planner reads `known()` for grounding; the executor calls `update()`
after every vision detection.
"""
from __future__ import annotations

import difflib
import re


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


class SpatialMemory:
    """label -> {xy:(x,y), last_seen:int, seen_count:int}. Fuzzy NL recall over labels."""

    def __init__(self) -> None:
        self._objs: dict[str, dict] = {}

    def update(self, label: str, xy: tuple[float, float], t: int = 0) -> None:
        label = (label or "").strip().lower()
        if not label:
            return
        if label in self._objs:
            o = self._objs[label]
            # exponential-ish smoothing toward the newest estimate
            ox, oy = o["xy"]
            o["xy"] = (0.5 * ox + 0.5 * xy[0], 0.5 * oy + 0.5 * xy[1])
            o["last_seen"] = t
            o["seen_count"] += 1
        else:
            self._objs[label] = {"xy": (float(xy[0]), float(xy[1])),
                                 "last_seen": t, "seen_count": 1}

    def known(self) -> dict[str, tuple[float, float]]:
        """label -> (x,y) for every remembered object (for planner grounding / HUD)."""
        return {k: v["xy"] for k, v in self._objs.items()}

    def recall(self, query: str) -> tuple[str, tuple[float, float]] | None:
        """Best fuzzy match of a natural-language query to a remembered object.

        Matches by token overlap first (color/shape words like 'green sphere'), then by
        difflib ratio over full labels. Returns (label, xy) or None if nothing is close.
        """
        if not self._objs:
            return None
        labels = list(self._objs.keys())
        q = (query or "").lower().strip()
        if not q:
            return None
        # exact / substring
        for lab in labels:
            if q == lab or q in lab or lab in q:
                return lab, self._objs[lab]["xy"]
        # token overlap (best shared color/shape words)
        qt = _tokens(q)
        best, best_score = None, 0
        for lab in labels:
            score = len(qt & _tokens(lab))
            if score > best_score:
                best, best_score = lab, score
        if best is not None and best_score > 0:
            return best, self._objs[best]["xy"]
        # difflib fallback
        m = difflib.get_close_matches(q, labels, n=1, cutoff=0.6)
        if m:
            return m[0], self._objs[m[0]]["xy"]
        return None

    def __len__(self) -> int:
        return len(self._objs)
