# Embodied navigation brain for the Unitree G1 (MuJoCo)

A chat-driven, vision-grounded whole-body controller for the G1 humanoid. You type a command;
a fast LLM parses intent; the robot walks to a target or performs a gesture — all whole-body
via a pretrained RL policy — while a live HUD shows what it **sees**, **thinks**, and where it
**is**. It remembers objects it has seen so it can go back to them later.

## Run it

```bash
# Interactive (headed) — opens the 3D viewer + the onboard-camera HUD, then chat in the terminal
E:\huminoid\.venv\Scripts\python.exe embodied\run_navigation_demo.py

# Headless acceptance gates (saves _gateA..D.png, prints PASS/FAIL)
E:\huminoid\.venv\Scripts\python.exe embodied\run_navigation_demo.py --script
```

The demo runs **until you quit** — type `quit`/`exit`/`q`, press `q` in the HUD window, or close
a window. (Optional safety cap: `--seconds 120`.)

### Viewer controls (native MuJoCo passive viewer)
- **scroll** = zoom in/out
- **left-drag** = orbit
- **right-drag** = pan

## What you can say (chat vocabulary)

| Intent | Examples | What happens |
|---|---|---|
| Walk to the stage | "go to the center of the stage", "walk to the red disk" | Navigates to the red disk at (2.5, 0). |
| Walk to a **remembered** object | "go to the green sphere" | Recalls the object's learned world position and walks there. |
| Find something not yet seen | "go to the blue crate" | Searches visually (does **not** invent a position). |
| Look / describe | "what do you see?" | Captions the onboard view (and learns objects into memory). |
| Look **back** at a remembered object | "look at the red circle" | Turns to face the remembered object, then re-captures it. |
| Gestures / skills | "raise your right hand", "wave", "bow", "nod", "clap", "celebrate", "point", "turn left/right" | Performs the closest whole-body skill. |
| Stand up / recover | "get up", "stand up", "recover" | Resets to a stable standing pose (recover-to-stand). |
| Anything unknown | "crawl on the floor", "do a backflip" | **Never refuses** — guesses the closest real skill. |

## How it stays grounded (no inventing)

The planner only acts on what the robot actually knows. Every command is planned against:
- the **real skill list** (the only skills it can pick),
- the **spatial memory** (objects seen + their world coords — the only places it can navigate to),
- the **current vision caption** (what it's seeing now).

Unseen targets trigger a *search*, not a made-up coordinate. See `DESIGN.md` for the closed loop.

## Where the compute runs (GPU split)

- **RL locomotion policy** → `torch` (CPU build) on the **CPU**. Small network, no CUDA needed.
- **MuJoCo rendering** → the **GPU** (AMD RX 580) via normal OpenGL.
- **Gemini** (3.5 Flash intent + Robotics-ER 1.6 vision) → **cloud API**, no local compute.

So the demo needs **no local GPU compute** — only OpenGL for the viewer. NVIDIA GR00T was ruled
out because it needs CUDA + ~24 GB VRAM (see `DESIGN.md`).

## Honest limitations

- **Stand-up** is a clean *state reset* to a balanced standing pose, not a physically-animated
  floor get-up (GMT ships no get-up reference clip).
- **Offline memory** (when the Gemini vision quota is exhausted) only learns the **red disk** —
  local OpenCV detection is red-only. Full multi-object recall needs Gemini-ER quota.
- **World-position estimates** use a single-shot bearing+range heuristic calibrated for the disk;
  for tall/large props it's approximate — good enough to navigate within arrival tolerance.

## Configuration

Put your key in `humanoid-gemini-brain/.env` (git-ignored, never committed):

```
GEMINI_API_KEY=...
```

Without a key (or once the free-tier quota is spent), the robust offline parser + local CV keep
the demo fully usable.
