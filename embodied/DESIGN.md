# Design — Embodied navigation brain (architecture of record)

> This is the living design doc. The full historical rationale (milestones M1–M3.5, the GMT
> pivot, model trade-offs) lives in the plan file; this doc is the distilled, current shape.

## Hierarchy

A VLM plans/reasons spatially; a learned RL policy walks and balances. An LLM cannot output
balancing joint trajectories — that belongs to the controller.

**Two demos share the same controller / perception / memory / HUD, differing only in the brain:**

### Classic demo — `run_navigation_demo.py` (3.5 Flash = brain, ER = eyes)
```
 camera ─▶ PERCEPTION ─▶ /camera,/feedback ─▶ PLANNER ─▶ Goal+Plan ─▶ EXECUTOR ─▶ NAV ─▶ CONTROLLER ─▶ MuJoCo
 proprio ─▶ (render+read)                  (3.5 Flash intent,   (applies goal)  (waypoint    (GMT 23-DoF
                                            grounded)                            →steer)      whole-body RL)
                              ▲                    ▲                    │
                            VISION ───────────▶ MEMORY ◀──────── writes detections (continuous)
                          (Gemini-ER 1.6)   (SpatialMemory)
```

### ER-boss demo — `run_orchestrator_demo.py` (Gemini-ER = boss, 3.5 Flash = skill sub-agent)
```
            live ego image + memory(known objects) + caption + command
                                   │
                                   ▼
                    ┌──────────────────────────────┐   no matching skill?
   command ────────▶│ GEMINI-ER 1.6  (orchestrator) │──────────────┐
                    │  decides ONE grounded action  │              ▼
                    └──────────────────────────────┘     ┌────────────────────────┐
                       │ Goal/skill   │ 429/error         │ 3.5 FLASH (sub-agent)   │
                       │              ▼ fallback           │ author skill spec ──────┼─▶ synthesize.py
                       │      planner.Brain (3.5/local)    └────────────────────────┘    (clip-safe .pkl)
                       ▼                                              │ new skill registered
                 EXECUTOR ─▶ NAV ─▶ GMT CONTROLLER ─▶ MuJoCo  ◀───────┘
            OVERLAY draws SEES / CMD / MEM / THINKS / PLAN / POS / BRAIN(ER|3.5|local)
```
**Memory feeds ER:** the orchestrator's prompt includes `known objects (label→world x,y)` from
`SpatialMemory`, so ER plans over what was actually seen and never invents coordinates. ER is
called per command (paid tier); on a per-minute rate-limit it cools down and the classic planner
takes over so the demo never freezes.

## Module map (`embodied/`)

| Module | Role |
|---|---|
| `controller/gmt_controller.py` | GMT pretrained 23-DoF whole-body policy on CPU; tracks a reference motion. `recover_to_stand()` = the stand-up / NaN-instability guard. |
| `messaging.py` | Typed dataclass messages + in-process pub/sub `Bus` (ROS2-shaped: `/camera`, `/feedback`, …). `Goal` carries `target_xy`/`skill`/`target_name`. |
| `perception.py` | Renders the 3rd-person + onboard (`render_ego`) camera; reads proprioception; publishes `/camera`,`/feedback`. |
| `vision.py` | `VisionBrain.look()` → Gemini Robotics-ER 1.6 caption + red-disk + multi-object detections; local OpenCV red-only fallback when offline. |
| `memory.py` | `SpatialMemory`: in-session `label -> {xy,last_seen,seen_count}`; fuzzy NL `recall()`; `known()` for grounding. |
| `planner.py` | `Brain.plan(command, scene, memory)` → `Goal`+`Plan`. Local parser first, then grounded Gemini 3.5 Flash, then never-refuse difflib guess. (Classic brain + the ER-boss fallback.) |
| `orchestrator.py` | **M3.6** `OrchestratorBrain.plan(command, scene, memory, image)` — Gemini-ER decides every command grounded in memory + image; delegates unknown skills to the 3.5 sub-agent; falls back to `planner.Brain` on ER rate-limit/error. |
| `skills_author.py` | **M3.6** 3.5 Flash skill sub-agent: `author_skill`/`build_skill` emit a clip-safe `SKILL_SPEC` over named upper-body DOFs (model-fallback across flash variants). |
| `synthesize.py` | Procedural GMT motions; `make_all()` (classic library) + `synthesize_from_spec()` (**M3.6** spec→motion, clipped to `SAFE_RANGE`). |
| `nav.py` | Waypoint → steering; proprio-based arrival check. |
| `run_navigation_demo.py` | **Classic demo.** `Executor` (applies goals per tick) + `build()` wiring + interactive/scripted entry. |
| `run_orchestrator_demo.py` | **M3.6 ER-boss demo.** Reuses `build`/`Executor`/overlay; swaps in `OrchestratorBrain`, passes the live ego image per command. |
| `test_models.py` | **M3.6** per-model harness (`--local`/`--flash`/`--er`/`--all`). |
| `overlay.py` | HUD draw (sees / cmd / mem / thinks / plan / pos / brain). |

## Message contracts (`messaging.py`)

- **`Goal`** `{kind, target_xy, skill, target_name, text}` — `kind ∈ go_to | go_to_visual | look | look_at | skill | idle`.
- **`Plan`** `{steps[], reasoning, current}` — the "what it's thinking" shown on the HUD.
- **`SceneView`** `{caption, targets{label->xy}}` — perception output; `targets` mirrors memory for the HUD.
- **`Feedback`** `{pos, yaw, height, upright, status}` — the deterministic success-check signal.

## Grounded planning (closed loop, no invented state)

`build()` creates **one** `SpatialMemory`, shared by:
- the **Executor**, which writes every vision detection into it (`_learn()` after each `vision.look()`), and
- **`brain.plan(cmd, scene, memory)`**, which reads it each tick.

So **perception → memory → planner** is closed. The planner's LLM call is fed a context block:

```
Available skills: [...]              # the only skills it may pick
Known objects (label -> world x,y): {...}   # the only places it may navigate to
Currently seeing: "<latest caption>"
```

Rules enforced (`planner.SYSTEM` + `_to_goal`):
- Never refuse — pick the closest **listed** skill (`_closest_skill` via difflib); `idle` only for empty/greeting.
- Navigation targets a **known** label or the stage landmark; never fabricate coordinates.
- A named-but-unseen target becomes a **visual search** (`go_to_visual`), not a guessed position.

**Planner-triggered look (grounded re-plan):** if a command plans to a `go_to_visual` for a
target the robot has *not* seen, the run loop first takes a fresh ER look (`_do_look`), learns
any detections into memory, then **re-plans** — so it can walk straight to the now-known
position instead of blindly searching. ER is still invoked only when seeing actually matters.

**Live HUD caption:** the `SEES` text is kept matching the live camera by a free, unlimited
local-CV refresh (`Executor.ambient_caption` → `VisionBrain.quick_look`) each render tick. A
rich ER caption (from an explicit "what do you see") is held for ~4 s before the ambient
refresh resumes, so the user can read it. ER quota is never spent on the ambient refresh.

## Detection → world position

`Executor._estimate_target(proprio, det)` turns one detection into a world `(x,y)`: horizontal
**bearing** from `cx` (camera FOV) + **range** from apparent `size`, projected from the robot's
pose. Used for any object, not just the disk. Single-shot heuristic — approximate but within
arrival tolerance.

## Safety: NaN-recover

If the GMT policy diverges, `QACC` goes NaN and rendering NaN geometry segfaults the process with
no Python traceback. `gmt_controller.step()` guards every tick: non-finite torque/state →
`recover_to_stand()` (reset to keyframe 0, zero action/history, hold the standing pose). The same
method backs the user-invokable "stand up" skill. `faulthandler` writes a C-stack to `_crash.log`
for anything that still escapes.

## Model choices

- **Gemini Robotics-ER 1.6** — perception/spatial reasoning (caption, pointing, object localization).
- **Gemini 3.5 Flash** — fast chat-intent parsing, escalated to only when the scene matters.
- **GMT (General Motion Tracking)** — pretrained 23-DoF G1 whole-body RL policy, runs on CPU.
- **NVIDIA GR00T — ruled out** for local use: needs CUDA + ~24 GB VRAM (machine has AMD RX 580 /
  4 GB). The `Brain` interface is swappable so a cloud-GPU GR00T could drop in later.

## Deferred

- **Memory persistence** (Voyager-adapted): disk-backed spatial store + episodic (goal, plan,
  outcome) log + a self-growing skill library. The in-session `SpatialMemory` leaves a clean seam.
- **Stairs / rough terrain** (M4): perceptive locomotion policy + terrain scene.
- **Real ROS2** (`rclpy`): only when moving to hardware — the bus is already topic-shaped.
