"""Direct G1 MuJoCo factory demo.

This is the clean v2 demo: no hidden gantry, no second robot, and no inherited
object executor. The selected part stays visible on the table until the G1 hand
reaches it, then it follows the G1 hand and is dropped into the selected bin.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np

from actions.g1_gesture_controller import G1GestureController
from brains.gemini_brain import GeminiBrain
from brains.mock_brain import MockBrain
from brains.schemas import Decision, SceneObject


ROOT = Path(__file__).resolve().parent
DEFAULT_SCENE = ROOT / "external" / "unitree_mujoco" / "unitree_robots" / "g1" / "g1_direct_factory_scene.xml"
DEFAULT_OUTPUT = ROOT / "output" / "g1_direct_factory"

OBJECT_CATALOG = {
    "red_part": {"label": "red defective part", "status": "defective", "color": "red"},
    "dark_part": {"label": "dark defective part", "status": "defective", "color": "dark"},
    "green_part": {"label": "green good part", "status": "good", "color": "green"},
}

SHORTCUTS = {
    ord("1"): "Pick up all defective parts and put them in the reject bin.",
    ord("2"): "Only pick the red defective part.",
    ord("3"): "Pick the dark defective part.",
    ord("4"): "Sort the green good part into the good bin.",
}


@dataclass
class Phase:
    name: str
    pose: str
    duration: float
    attach_to_hand: bool = False
    drop_at_end: bool = False


class DirectG1World:
    def __init__(self, scene_path: Path, width: int, height: int):
        self.scene_path = scene_path
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        mujoco.mj_forward(self.model, self.data)
        self.g1 = G1GestureController(self.model, self.data)
        self.hand_body_id = self.model.body("right_wrist_roll_rubber_hand").id
        self.home_positions = {object_id: self.object_position(object_id) for object_id in OBJECT_CATALOG}
        self.settled: dict[str, str] = {}
        self.active_object = ""
        self.active_destination = "reject_bin"
        self.phases: list[Phase] = []
        self.phase_index = 0
        self.phase_elapsed = 0.0
        self._started_phase = ""
        self._celebrated = False
        self.pin_objects()

    @property
    def busy(self) -> bool:
        return bool(self.phases)

    def close(self) -> None:
        self.renderer.close()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.g1 = G1GestureController(self.model, self.data)
        self.home_positions = {object_id: self.object_position(object_id) for object_id in OBJECT_CATALOG}
        self.settled.clear()
        self.active_object = ""
        self.phases.clear()
        self.phase_index = 0
        self.phase_elapsed = 0.0
        self._started_phase = ""
        self._celebrated = False
        self.pin_objects()

    def start_decision(self, decision: Decision) -> str:
        if self.busy:
            return "G1 is already executing a task."
        if not decision.is_executable():
            return f"No executable action: {decision.action}"
        if decision.target_object not in OBJECT_CATALOG:
            return f"Unknown target object: {decision.target_object}"
        if decision.target_object in self.settled:
            return f"{decision.target_object} is already sorted."

        self.active_object = decision.target_object
        self.active_destination = decision.destination
        reach_pose = {
            "red_part": "reach_red",
            "dark_part": "reach_dark",
            "green_part": "reach_green",
        }.get(decision.target_object, "inspect")
        place_pose = "good_bin" if decision.destination == "good_bin" else "reject"
        self.phases = [
            Phase("look at table", "inspect", 0.75),
            Phase("reach part", reach_pose, 1.15),
            Phase("grasp part", reach_pose, 0.35, attach_to_hand=True),
            Phase("carry to bin", place_pose, 1.25, attach_to_hand=True),
            Phase("release", place_pose, 0.25, drop_at_end=True),
            Phase("return home", "home", 0.80),
        ]
        self.phase_index = 0
        self.phase_elapsed = 0.0
        self._started_phase = ""
        return f"G1 started {decision.target_object} -> {decision.destination}."

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self._step_task()
            self.g1.step()
            mujoco.mj_step(self.model, self.data)
            self.pin_objects(exclude_active=True)
            if self.active_object and self._current_phase_attach():
                self.set_object_position(self.active_object, self.hand_object_position())

    def _step_task(self) -> None:
        if not self.phases:
            self.pin_objects()
            return
        phase = self.phases[self.phase_index]
        if self._started_phase != phase.name:
            self.g1.start_pose(phase.pose, phase.duration)
            self._started_phase = phase.name
            self.phase_elapsed = 0.0

        if phase.attach_to_hand and self.active_object:
            self.set_object_position(self.active_object, self.hand_object_position())

        self.phase_elapsed += float(self.model.opt.timestep)
        if self.phase_elapsed < phase.duration:
            return

        if phase.drop_at_end and self.active_object:
            self.set_object_position(self.active_object, self.bin_position(self.active_destination, self.active_object))
            self.settled[self.active_object] = self.active_destination

        self.phase_index += 1
        self.phase_elapsed = 0.0
        self._started_phase = ""
        if self.phase_index >= len(self.phases):
            self.active_object = ""
            self.phases.clear()
            self.phase_index = 0
            self.pin_objects()

    def _current_phase_attach(self) -> bool:
        return bool(self.phases and self.phases[self.phase_index].attach_to_hand)

    def maybe_celebrate_done(self) -> None:
        if not self._celebrated and self.unsorted_defective_count() == 0:
            self.g1.start_pose("celebrate_done", 1.2)
            self._celebrated = True

    def frame_bgr(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera="factory_cam")
        return cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)

    def scene_hint(self) -> dict[str, Any]:
        objects = []
        for object_id, meta in OBJECT_CATALOG.items():
            if object_id in self.settled:
                continue
            pos = self.object_position(object_id)
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
            "scene": "G1 direct MuJoCo inspection cell",
            "bins": {
                "reject_bin": "red bin on the table; defective parts go here",
                "good_bin": "blue bin near the G1 hand; accepted parts go here",
            },
            "objects": objects,
        }

    def unsorted_defective_count(self) -> int:
        return sum(
            1
            for object_id, meta in OBJECT_CATALOG.items()
            if meta["status"] == "defective" and object_id not in self.settled
        )

    def object_position(self, object_id: str) -> np.ndarray:
        joint_id = self.model.joint(f"{object_id}_free").id
        qpos_adr = self.model.jnt_qposadr[joint_id]
        return self.data.qpos[qpos_adr : qpos_adr + 3].copy()

    def set_object_position(self, object_id: str, pos: np.ndarray) -> None:
        joint_id = self.model.joint(f"{object_id}_free").id
        qpos_adr = self.model.jnt_qposadr[joint_id]
        qvel_adr = self.model.jnt_dofadr[joint_id]
        self.data.qpos[qpos_adr : qpos_adr + 3] = pos
        self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qvel[qvel_adr : qvel_adr + 6] = 0.0

    def hand_object_position(self) -> np.ndarray:
        return self.data.xpos[self.hand_body_id].copy() + np.array([0.0, 0.0, -0.025])

    def bin_position(self, destination: str, object_id: str) -> np.ndarray:
        body_id = self.model.body(destination).id
        base = self.data.xpos[body_id].copy()
        offsets = {
            "red_part": np.array([0.00, 0.035, 0.0]),
            "dark_part": np.array([0.00, -0.035, 0.0]),
            "green_part": np.array([0.00, 0.00, 0.0]),
        }
        return np.array([base[0], base[1], 1.02], dtype=float) + offsets.get(object_id, 0.0)

    def pin_objects(self, exclude_active: bool = False) -> None:
        for object_id in OBJECT_CATALOG:
            if exclude_active and object_id == self.active_object:
                continue
            if object_id in self.settled:
                self.set_object_position(object_id, self.bin_position(self.settled[object_id], object_id))
            else:
                self.set_object_position(object_id, self.home_positions[object_id])


def build_brain(name: str, model: str):
    if name == "mock":
        return MockBrain()
    if name == "gemini":
        return GeminiBrain(model=model)
    raise ValueError(name)


def annotate(
    frame: np.ndarray,
    instruction: str,
    decision: Decision | None,
    status: str,
    brain_name: str,
    world: DirectG1World,
) -> np.ndarray:
    out = frame.copy()
    overlay = out.copy()
    y0 = max(18, out.shape[0] - 184)
    cv2.rectangle(overlay, (18, y0), (out.shape[1] - 18, out.shape[0] - 18), (20, 24, 28), -1)
    cv2.addWeighted(overlay, 0.72, out, 0.28, 0, out)

    def put(text: str, row: int, color=(255, 255, 255)) -> None:
        cv2.putText(out, text, (34, y0 + row), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)

    put(f"Instruction: {instruction[:110]}", 30)
    put(f"Brain: {brain_name} | G1 pose: {world.g1.current_pose} | active: {world.active_object or '-'}", 58, (230, 230, 120))
    if decision:
        put(f"Decision: {decision.action} target={decision.target_object or '-'} dest={decision.destination}", 86, (120, 220, 255))
        put(f"Reason: {decision.reasoning[:120]}", 114, (210, 235, 210))
    else:
        put("Decision: waiting", 86, (120, 220, 255))
    put(f"Status: {status} | remaining defects: {world.unsorted_defective_count()}", 140, (230, 230, 120))
    put("Keys: 1 all defects | 2 red | 3 dark | 4 green | r reset | q quit", 166, (220, 220, 220))
    return out


def run(args: argparse.Namespace) -> int:
    world = DirectG1World(Path(args.scene), args.width, args.height)
    brain = build_brain(args.brain, args.gemini_model)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    instruction = args.instruction
    decision: Decision | None = None
    status = "ready"
    records: list[dict[str, Any]] = []
    next_brain_time = 0.0
    frame_idx = 0
    started = time.time()

    print(f"Scene: {world.scene_path}")
    print(f"Brain: {args.brain}")
    print(f"Instruction: {instruction}")

    try:
        while frame_idx < args.max_frames:
            world.step(args.physics_steps_per_frame)
            now = time.time()

            if not world.busy and now >= next_brain_time:
                if world.unsorted_defective_count() == 0 and "defect" in instruction.lower():
                    status = "complete: no defective objects remain"
                    world.maybe_celebrate_done()
                    if args.stop_when_done:
                        break
                else:
                    frame = world.frame_bgr()
                    hint = world.scene_hint()
                    decision = brain.decide(frame, instruction, hint)
                    status = world.start_decision(decision)
                    next_brain_time = now + args.brain_interval
                    records.append(
                        {
                            "time_s": round(now - started, 3),
                            "decision": decision.__dict__,
                            "status": status,
                            "scene_hint": hint,
                        }
                    )
                    print(f"[brain] {decision.action} {decision.target_object} -> {decision.destination}")
                    print(f"        {status}")

            frame = world.frame_bgr()
            annotated = annotate(frame, instruction, decision, status, args.brain, world)

            if args.save_frames and frame_idx % args.save_every == 0:
                cv2.imwrite(str(output_dir / f"frame_{frame_idx:05d}.jpg"), annotated)
            if not args.no_cv_window:
                cv2.imshow("Direct G1 Factory Demo", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    world.reset()
                    decision = None
                    status = "reset"
                    next_brain_time = 0.0
                elif key in SHORTCUTS:
                    instruction = SHORTCUTS[key]
                    status = f"new command: {instruction}"
                    next_brain_time = 0.0

            frame_idx += 1
    finally:
        if not args.no_cv_window:
            cv2.destroyAllWindows()
        (output_dir / "decision_log.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        world.close()

    print(f"Finished: {status}")
    print(f"Output: {output_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain", choices=["mock", "gemini"], default="mock")
    parser.add_argument("--gemini-model", default="gemini-2.5-flash")
    parser.add_argument("--instruction", default="Pick up all defective parts and put them in the reject bin.")
    parser.add_argument("--scene", default=str(DEFAULT_SCENE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-frames", type=int, default=1400)
    parser.add_argument("--brain-interval", type=float, default=0.4)
    parser.add_argument("--physics-steps-per-frame", type=int, default=2)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--no-cv-window", action="store_true")
    parser.add_argument("--stop-when-done", action="store_true", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
