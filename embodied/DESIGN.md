# Design — Embodied navigation brain (architecture of record)

> This is the living design doc. The full historical rationale (milestones M1–M3.5, the GMT
> pivot, model trade-offs) lives in the plan file; this doc is the distilled, current shape.

## Hierarchy

A VLM plans/reasons spatially; a learned RL policy walks and balances. An LLM cannot output
balancing joint trajectories — that belongs to the controller.

```
 camera ─▶ PERCEPTION ─▶ /camera,/feedback ─▶ PLANNER ─▶ Goal+Plan ─▶ EXECUTOR ─▶ NAV ─▶ CONTROLLER ─▶ MuJoCo
 proprio ─▶ (render+read)                     (intent,           (applies goal)  (waypoint    (GMT 23-DoF
                                               grounded)                          →steer)      whole-body RL)
                              ▲                    ▲                    │
                              │                    │                    ▼
                            VISION ───────────▶ MEMORY ◀──────── writes detections
                          (Gemini-ER 1.6)   (SpatialMemory)     (label -> world x,y)
            OVERLAY draws SEES / CMD / MEM / THINKS / PLAN / POS on the onboard frame
```

## Module map (`embodied/`)

| Module | Role |
|---|---|
| `controller/gmt_controller.py` | GMT pretrained 23-DoF whole-body policy on CPU; tracks a reference motion. `recover_to_stand()` = the stand-up / NaN-instability guard. |
| `messaging.py` | Typed dataclass messages + in-process pub/sub `Bus` (ROS2-shaped: `/camera`, `/feedback`, …). `Goal` carries `target_xy`/`skill`/`target_name`. |
| `perception.py` | Renders the 3rd-person + onboard (`render_ego`) camera; reads proprioception; publishes `/camera`,`/feedback`. |
| `vision.py` | `VisionBrain.look()` → Gemini Robotics-ER 1.6 caption + red-disk + multi-object detections; local OpenCV red-only fallback when offline. |
| `memory.py` | `SpatialMemory`: in-session `label -> {xy,last_seen,seen_count}`; fuzzy NL `recall()`; `known()` for grounding. |
| `planner.py` | `Brain.plan(command, scene, memory)` → `Goal`+`Plan`. Local parser first, then grounded Gemini 3.5 Flash, then never-refuse difflib guess. |
| `nav.py` | Waypoint → steering; proprio-based arrival check. |
| `run_navigation_demo.py` | `Executor` (applies goals per tick) + `build()` wiring + interactive/scripted entry points. |
| `overlay.py` | HUD draw (sees / cmd / mem / thinks / plan / pos). |

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
