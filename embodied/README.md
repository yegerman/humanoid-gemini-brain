# Embodied navigation brain for the Unitree G1 (MuJoCo)

A chat-driven, vision-grounded whole-body controller for the G1 humanoid. You type a command;
a fast LLM parses intent; the robot walks to a target or performs a gesture — all whole-body
via a pretrained RL policy — while a live HUD shows what it **sees**, **thinks**, and where it
**is**. It remembers objects it has seen so it can go back to them later.

## Run it

Two demos share the same robot, scene, memory and HUD — they differ only in the brain.

**Quick launch (from the repo root, no venv path needed):**
```powershell
.\demo-er.bat            # ER-boss demo (default --er-secs 30)
.\demo.bat               # classic demo
.\demo.bat --script      # headless gates
.\demo-er.bat --er-secs 60   # spend even less on ER
```
(`.ps1` versions exist too: `.\demo-er.ps1`.)

Or call Python directly:
```bash
# Classic demo — Gemini 3.5 Flash parses intent, Gemini-ER is the eyes
E:\huminoid\.venv\Scripts\python.exe embodied\run_navigation_demo.py

# ER-boss demo (M3.6) — Gemini-ER decides every command from the live camera + memory,
# and delegates unknown skills to a 3.5 Flash sub-agent that AUTHORS new motions on the fly
E:\huminoid\.venv\Scripts\python.exe embodied\run_orchestrator_demo.py

# Headless acceptance gates (classic; saves _gateA..D.png, prints PASS/FAIL)
E:\huminoid\.venv\Scripts\python.exe embodied\run_navigation_demo.py --script

# Per-model tests (validate each model independently)
E:\huminoid\.venv\Scripts\python.exe embodied\test_models.py --all   # or --local / --flash / --er
```

In the ER-boss demo the HUD shows `BRAIN: ER | 3.5 | local` (who decided), and a novel command
like "salute" makes ER ask the 3.5 sub-agent to author + synthesize a new skill, which the robot
then performs.

**Cost control:** ER (the expensive image call) sees + decides at most once every **30 s** by
default (`--er-secs 30`); between those, the cheap local/3.5 planner handles commands, grounded in
memory + the last ER caption. Use `--er-secs 0` for ER on every command, or a larger number to
spend even less. If ER is momentarily rate-limited, the classic planner takes over automatically.

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

**Continuous perception:** every frame the robot updates the `SEES` caption AND feeds what it
sees into spatial memory — for free (local OpenCV, no API). A rich Gemini-ER caption from
"what do you see" is held briefly so you can read it. If you ask to go to something it hasn't
seen, it takes a fresh look first, then plans to walk there.

For automatic *rich* (multi-object) perception on a timer, add `--look-secs N` (e.g. `30`). This
spends Gemini-ER quota every N seconds, so it's **off by default** (the free continuous local
perception above always runs regardless).

## Where the compute runs (GPU split)

- **RL locomotion policy** → `torch` (CPU build) on the **CPU**. Small network, no CUDA needed.
- **MuJoCo rendering** → the **GPU** (AMD RX 580) via normal OpenGL.
- **Gemini** (3.5 Flash intent + Robotics-ER 1.6 vision) → **cloud API**, no local compute.

So the demo needs **no local GPU compute** — only OpenGL for the viewer. NVIDIA GR00T was ruled
out because it needs CUDA + ~24 GB VRAM (see `DESIGN.md`).

## Honest limitations

- **Stand-up** is a clean *state reset* to a balanced standing pose, not a physically-animated
  floor get-up (GMT ships no get-up reference clip).
- **Offline vision** (when Gemini-ER quota is unavailable) uses a free local **multi-color HSV**
  detector: it names colored props (red/orange/yellow/green/blue/purple) with a coarse
  aspect-ratio shape guess, so the caption + memory track the camera without ER. Precise shape /
  rich descriptions still need ER (which now retries after a ~60 s cooldown rather than staying
  disabled for the session).
- **World-position estimates** cast a ray from the object's base pixel to the ground plane using
  the robot's real ego-camera pose — accurate for any size/shape since props rest on the floor
  (e.g. green sphere estimated at (3.6, 1.47) vs true (3.6, 1.4)). Far/edge/occluded objects are
  less precise. Memory is keyed by color so repeated sightings merge into one stable entry.

## Configuration

Put your key in `humanoid-gemini-brain/.env` (git-ignored, never committed):

```
GEMINI_API_KEY=...
```

Without a key (or once the free-tier quota is spent), the robust offline parser + local CV keep
the demo fully usable.
