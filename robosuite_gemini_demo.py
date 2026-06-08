"""Gemini-controlled MuJoCo factory inspection demo.

The name keeps the planned robosuite direction, but the default backend is a
bundled MuJoCo scene that runs without extra downloads. If robosuite is added
later, it should implement the same Brain -> Decision -> ActionExecutor flow.
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np

from actions.g1_gesture_controller import G1GestureController
from actions.mujoco_executor import MuJoCoFactoryExecutor
from brains.gemini_brain import GeminiBrain
from brains.mock_brain import MockBrain
from brains.schemas import Decision, SceneObject


ROOT = Path(__file__).resolve().parent
DEFAULT_G1_SCENE = ROOT / "external" / "unitree_mujoco" / "unitree_robots" / "g1" / "g1_realistic_factory_scene.xml"
FALLBACK_G1_SCENE = ROOT / "external" / "unitree_mujoco" / "unitree_robots" / "g1" / "gemini_g1_factory_scene.xml"
DEFAULT_SCENE = DEFAULT_G1_SCENE if DEFAULT_G1_SCENE.exists() else FALLBACK_G1_SCENE
DEFAULT_OUTPUT = ROOT / "output" / "gemini_factory"

OBJECT_CATALOG = {
    "red_part": {
        "label": "red defect part",
        "status": "defective",
        "color": "red",
    },
    "dark_part": {
        "label": "dark discolor defect part",
        "status": "defective",
        "color": "dark",
    },
    "green_part": {
        "label": "green good part",
        "status": "good",
        "color": "green",
    },
}


class FactoryWorld:
    def __init__(
        self,
        scene_path: Path,
        width: int = 960,
        height: int = 720,
        enable_g1: bool = True,
    ) -> None:
        self.scene_path = scene_path
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.width = width
        self.height = height
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        mujoco.mj_forward(self.model, self.data)
        self.executor = MuJoCoFactoryExecutor(self.model, self.data, list(OBJECT_CATALOG))
        self.g1 = G1GestureController(self.model, self.data) if enable_g1 else None
        self._celebrated_done = False

    def close(self) -> None:
        self.renderer.close()

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self.executor.step()
            if self.g1 is not None:
                self.g1.step()
            mujoco.mj_step(self.model, self.data)
            self.executor.stabilize_after_physics()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.executor = MuJoCoFactoryExecutor(self.model, self.data, list(OBJECT_CATALOG))
        if self.g1 is not None:
            self.g1 = G1GestureController(self.model, self.data)
        self._celebrated_done = False

    def start_g1_for_decision(self, decision: Decision) -> None:
        if self.g1 is not None and decision.is_executable():
            self.g1.start_task_sequence(decision.target_object, decision.destination)

    def g1_home(self) -> None:
        if self.g1 is not None:
            self.g1.home()

    def maybe_celebrate_done(self) -> None:
        if self.g1 is not None and not self._celebrated_done:
            self.g1.celebrate()
            self._celebrated_done = True

    def g1_pose(self) -> str:
        if self.g1 is None:
            return "off"
        return self.g1.current_pose

    def g1_busy(self) -> bool:
        return bool(self.g1 is not None and self.g1.busy)

    def frame_bgr(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera="factory_cam")
        frame_rgb = self.renderer.render()
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    def scene_hint(self) -> dict[str, Any]:
        objects: list[dict[str, Any]] = []
        for object_id, meta in OBJECT_CATALOG.items():
            if object_id in self.executor.settled_objects:
                continue
            pos = self.executor.object_position(object_id)
            objects.append(
                SceneObject(
                    object_id=object_id,
                    label=meta["label"],
                    status=meta["status"],
                    color=meta["color"],
                    position=(float(pos[0]), float(pos[1]), float(pos[2])),
                ).to_prompt_dict()
            )
        return {
            "scene": "MuJoCo factory inspection workcell",
            "camera": "factory_cam",
            "bins": {
                "reject_bin": "red bin at robot right; defective parts go here",
                "good_bin": "blue bin at robot left; accepted parts go here",
            },
            "objects": objects,
        }

    def unsorted_defective_count(self) -> int:
        return sum(
            1
            for object_id, meta in OBJECT_CATALOG.items()
            if meta["status"] == "defective" and object_id not in self.executor.settled_objects
        )


def build_brain(name: str, model: str):
    if name == "mock":
        return MockBrain()
    if name == "gemini":
        return GeminiBrain(model=model)
    raise ValueError(f"Unknown brain: {name}")


COMMAND_SHORTCUTS = {
    ord("1"): "Pick up all defective parts and put them in the reject bin.",
    ord("2"): "Only pick the red defective part.",
    ord("3"): "Pick the dark discolored part.",
    ord("4"): "Sort the good green part into the good bin.",
}


def start_chat_thread(initial_instruction: str) -> queue.Queue[str]:
    commands: queue.Queue[str] = queue.Queue()

    def read_commands() -> None:
        print("")
        print("Chat channel ready. Type a new robot instruction and press Enter.")
        print("Examples: 'Only pick the red defective part' or 'Sort the good green part into the good bin'")
        print("Type 'quit' to stop.")
        print("")
        while True:
            try:
                text = input("robot> ").strip()
            except EOFError:
                return
            if text:
                commands.put(text)
            if text.lower() in {"quit", "exit", "stop"}:
                return

    commands.put(initial_instruction)
    thread = threading.Thread(target=read_commands, daemon=True)
    thread.start()
    return commands


def annotate_frame(
    frame: np.ndarray,
    instruction: str,
    decision: Decision | None,
    status: str,
    remaining: int,
    brain_mode: str,
    g1_pose: str,
    desktop_ui: bool,
) -> np.ndarray:
    out = frame.copy()
    overlay = out.copy()
    y0 = max(18, out.shape[0] - 188)
    cv2.rectangle(overlay, (18, y0), (out.shape[1] - 18, out.shape[0] - 18), (20, 24, 28), -1)
    cv2.addWeighted(overlay, 0.72, out, 0.28, 0, out)

    def put(line: str, row: int, color=(255, 255, 255)) -> None:
        cv2.putText(out, line, (34, y0 + row), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)

    put(f"Instruction: {instruction[:110]}", 30)
    put(f"Brain: {brain_mode} | G1: {g1_pose}", 58, (230, 230, 120))
    if decision:
        put(
            f"Decision: {decision.action} target={decision.target_object or '-'} "
            f"dest={decision.destination} conf={decision.confidence:.2f}",
            86,
            (120, 220, 255),
        )
        put(f"Reason: {decision.reasoning[:120]}", 114, (210, 235, 210))
    else:
        put("Decision: waiting for first instruction", 86, (120, 220, 255))
    put(f"Status: {status} | remaining defects: {remaining}", 140, (230, 230, 120))
    if desktop_ui:
        put("Keys: 1 all defects | 2 red | 3 dark | 4 green | t type | r reset | h home | q quit", 166, (220, 220, 220))
    return out


def save_decision_log(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def run_demo(args: argparse.Namespace) -> int:
    enable_g1 = args.g1 and not args.no_g1
    world = FactoryWorld(Path(args.scene), width=args.width, height=args.height, enable_g1=enable_g1)
    brain = build_brain(args.brain, args.gemini_model)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_writer = None
    if args.record:
        video_path = output_dir / "gemini_factory_demo.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (args.width, args.height))

    viewer = None
    if not args.no_viewer:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(world.model, world.data)

    decision: Decision | None = None
    instruction = args.instruction
    chat_commands = start_chat_thread(instruction) if args.chat else None
    status = "starting"
    records: list[dict[str, Any]] = []
    next_brain_time = 0.0
    start = time.time()
    frame_idx = 0

    print(f"Scene: {world.scene_path}")
    print(f"Brain: {args.brain}")
    print(f"G1 gestures: {'on' if enable_g1 and world.g1 is not None and world.g1.enabled else 'off'}")
    print(f"Instruction: {instruction}")
    print("Press Ctrl+C to stop.")

    try:
        while frame_idx < args.max_frames:
            now = time.time()
            world.step(args.physics_steps_per_frame)

            if chat_commands is not None:
                while not chat_commands.empty():
                    text = chat_commands.get()
                    if text.lower() in {"quit", "exit", "stop"}:
                        status = "stopped from chat"
                        frame_idx = args.max_frames
                        break
                    instruction = text
                    status = f"new instruction: {instruction}"
                    if args.reset_on_command:
                        world.reset()
                    next_brain_time = 0.0
                    print(f"[chat] {instruction}")

            if not world.executor.busy and not world.g1_busy() and now >= next_brain_time:
                if world.unsorted_defective_count() == 0 and "defect" in instruction.lower():
                    status = "complete: no defective objects remain"
                    world.maybe_celebrate_done()
                    if args.stop_when_done and not args.chat:
                        break
                else:
                    frame = world.frame_bgr()
                    hint = world.scene_hint()
                    decision = brain.decide(frame, instruction, hint)
                    exec_status = world.executor.execute(decision)
                    if exec_status.started:
                        world.start_g1_for_decision(decision)
                    status = exec_status.message
                    next_brain_time = now + args.brain_interval
                    records.append(
                        {
                            "time_s": round(now - start, 3),
                            "decision": decision.__dict__,
                            "execution": exec_status.__dict__,
                            "scene_hint": hint,
                        }
                    )
                    print(f"[brain] {decision.action} {decision.target_object} -> {decision.destination}")
                    print(f"        {decision.reasoning}")

            frame = world.frame_bgr()
            annotated = annotate_frame(
                frame,
                instruction,
                decision,
                status,
                world.unsorted_defective_count(),
                args.brain,
                world.g1_pose(),
                args.desktop_ui,
            )
            if args.save_frames and frame_idx % args.save_every == 0:
                cv2.imwrite(str(output_dir / f"frame_{frame_idx:05d}.jpg"), annotated)
            if video_writer:
                video_writer.write(annotated)
            if viewer:
                viewer.sync()
            if not args.no_cv_window:
                cv2.imshow("Gemini MuJoCo Factory Demo", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key in COMMAND_SHORTCUTS:
                    instruction = COMMAND_SHORTCUTS[key]
                    status = f"keyboard command: {instruction}"
                    if args.reset_on_command:
                        world.reset()
                    next_brain_time = 0.0
                    print(f"[key] {instruction}")
                elif key == ord("r"):
                    world.reset()
                    decision = None
                    status = "scene reset"
                    next_brain_time = 0.0
                    print("[key] reset scene")
                elif key == ord("h"):
                    world.g1_home()
                    status = "G1 returning home"
                    print("[key] G1 home")
                elif key == ord("t"):
                    typed = input("robot> ").strip()
                    if typed:
                        instruction = typed
                        status = f"typed command: {instruction}"
                        if args.reset_on_command:
                            world.reset()
                        next_brain_time = 0.0
                        print(f"[typed] {instruction}")

            frame_idx += 1

    except KeyboardInterrupt:
        status = "stopped by user"
    finally:
        if video_writer:
            video_writer.release()
        if viewer:
            viewer.close()
        if not args.no_cv_window:
            cv2.destroyAllWindows()
        save_decision_log(output_dir / "decision_log.json", records)
        world.close()

    print(f"Finished: {status}")
    print(f"Decision log: {output_dir / 'decision_log.json'}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain", choices=["mock", "gemini"], default="mock")
    parser.add_argument("--gemini-model", default="gemini-2.5-flash")
    parser.add_argument(
        "--instruction",
        default="Pick up all defective parts and put them in the reject bin.",
    )
    parser.add_argument("--scene", default=str(DEFAULT_SCENE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-frames", type=int, default=1200)
    parser.add_argument("--brain-interval", type=float, default=1.0)
    parser.add_argument("--physics-steps-per-frame", type=int, default=2)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--no-cv-window", action="store_true")
    parser.add_argument("--desktop-ui", action="store_true", help="Enable keyboard command overlay in the OpenCV window.")
    parser.add_argument("--g1", action="store_true", default=True, help="Enable visible G1 gesture control.")
    parser.add_argument("--no-g1", action="store_true", help="Disable visible G1 gesture control.")
    parser.add_argument("--reset-on-command", action="store_true", help="Reset object positions before each new command.")
    parser.add_argument("--chat", action="store_true", help="Read live robot instructions from the terminal.")
    parser.add_argument("--stop-when-done", action="store_true", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_demo(parse_args()))
