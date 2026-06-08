# Gemini-Controlled MuJoCo Factory Robot Demo

This POC replaces the YOLO training pipeline with a hosted Gemini vision brain.
You type a natural-language instruction, Gemini or the offline mock brain chooses
the next object/action, and a local MuJoCo action executor performs a reliable
pick-and-place in a factory-style inspection cell.

## What Runs Now

The default implementation now uses the official Unitree G1 MuJoCo model inside
a richer factory-style inspection cell:

```text
external/unitree_mujoco/unitree_robots/g1/g1_realistic_factory_scene.xml
```

The older lightweight scenes are still available for fallback:

```text
scene/gemini_factory_scene.xml
external/unitree_mujoco/unitree_robots/g1/gemini_g1_factory_scene.xml
```

It includes:

- official Unitree G1 humanoid
- metal work table
- conveyor lane with rollers
- red reject bin
- blue accepted-parts bin
- overhead gantry gripper
- red defective part
- dark defective part
- green good part
- walls, light bars, shelves, crates, safety tape, and tower light props

The code is structured so robosuite `PickPlace` can be added later behind the
same brain/action interface. robosuite is not required for the working demo.

## Architecture

```text
User instruction
      |
      v
Camera frame ---> Brain ---> Decision JSON ---> Action executor ---> MuJoCo scene
                 mock          target/action       pick/place         objects move
                 Gemini
                                      |
                                      v
                             G1 gesture controller
                             inspect/reach/reject/home
```

Gemini decides **what** should happen. The local executor decides **how** to move
objects safely and reliably. The G1 gesture controller makes the humanoid visibly
respond to the task without requiring ROS2, balance control, or real hand grasping.

## Files

```text
robosuite_gemini_demo.py          main entry point
brains/schemas.py                 Decision and SceneObject schemas
brains/mock_brain.py              offline deterministic brain
brains/gemini_brain.py            hosted Gemini adapter
actions/mujoco_executor.py        reliable pick/place skills
actions/g1_gesture_controller.py  visible G1 inspect/reach/reject gestures
scene/gemini_factory_scene.xml    bundled MuJoCo factory cell
external/unitree_mujoco/unitree_robots/g1/g1_realistic_factory_scene.xml
docs/Gemini_MuJoCo_POC_Architecture.docx
```

## Run Offline

No API key required:

```powershell
cd E:\huminoid
.venv\Scripts\python.exe robosuite_gemini_demo.py --brain mock --g1 --desktop-ui
```

Run with the explicit G1 realistic scene:

```powershell
.venv\Scripts\python.exe robosuite_gemini_demo.py --brain mock --g1 --desktop-ui --scene E:\huminoid\external\unitree_mujoco\unitree_robots\g1\g1_realistic_factory_scene.xml
```

Run with a direct chat channel:

```powershell
.venv\Scripts\python.exe robosuite_gemini_demo.py --brain mock --chat --g1 --desktop-ui --scene E:\huminoid\external\unitree_mujoco\unitree_robots\g1\g1_realistic_factory_scene.xml
```

Desktop UI keys:

```text
1 = pick all defective parts
2 = only pick the red defective part
3 = pick the dark discolored part
4 = sort the good green part into the good bin
t = type a custom instruction in the terminal
r = reset the scene
h = send G1 home
q = quit
```

While the demo is running, type commands into the terminal:

```text
robot> Only pick the red defective part
robot> Pick the dark discolored part
robot> Sort the good green part into the good bin
robot> quit
```

Headless verification:

```powershell
.venv\Scripts\python.exe robosuite_gemini_demo.py --brain mock --g1 --desktop-ui --no-viewer --no-cv-window --save-frames --output E:\huminoid\output\g1_realistic_ui_smoke
```

## Run With Gemini

Install requirements already added to the venv:

```powershell
.venv\Scripts\python.exe -m pip install google-genai python-dotenv
```

Set an API key:

```powershell
$env:GEMINI_API_KEY = "your_key_here"
```

Run:

```powershell
.venv\Scripts\python.exe robosuite_gemini_demo.py --brain gemini --g1 --desktop-ui --instruction "Pick up all defective parts and put them in the reject bin."
```

## Useful Instructions To Try

```text
Pick up all defective parts and put them in the reject bin.
Only pick the red defective part.
Ignore green parts.
Pick the dark discolored part.
Sort the good green part into the good bin.
```

## Current Hardware Reality

This PC has only Python 3.14 installed. robosuite currently pulls dependencies
that expect older Python/NumPy wheels, so installing robosuite in this venv tried
to compile NumPy and failed without Visual Studio build tools. The implemented
fallback uses pure MuJoCo and runs now.

Recommended later robosuite path:

```powershell
py -3.11 -m venv C:\huminoid\.venv-robosuite
C:\huminoid\.venv-robosuite\Scripts\python.exe -m pip install robosuite google-genai opencv-python
```

## Verification Status

Offline smoke test passed:

```text
[brain] pick_and_place red_part -> reject_bin
[brain] pick_and_place dark_part -> reject_bin
Finished: complete: no defective objects remain
```

Saved frames and decision logs are written under:

```text
E:\huminoid\output\*
```

The active project path is now:

```text
E:\huminoid
```
