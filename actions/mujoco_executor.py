"""Reliable pick/place skills for the bundled MuJoCo factory scene."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from brains.schemas import Decision


@dataclass
class ExecutionStatus:
    started: bool
    completed: bool
    message: str


class MuJoCoFactoryExecutor:
    """Moves inspection parts reliably while the G1 provides visible action."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, object_ids: list[str]):
        self.model = model
        self.data = data
        self.object_ids = object_ids
        self.bin_positions = {
            "reject_bin": self._bin_drop_position("reject_bin"),
            "good_bin": self._bin_drop_position("good_bin"),
        }
        self.home = np.array([0.45, 0.0, 1.35], dtype=float)
        self._mocap_id = self.model.body("gantry_gripper").mocapid[0]
        self._active_plan: list[np.ndarray] = []
        self._active_object = ""
        self._active_destination = "reject_bin"
        self._last_active_pos: np.ndarray | None = None
        self._step_idx = 0
        self._settled_objects: set[str] = set()
        self._settled_destinations: dict[str, str] = {}
        self._home_positions = {
            object_id: self.object_position(object_id)
            for object_id in object_ids
        }
        self.set_gripper(self.home)
        self.pin_all_objects()

    @property
    def busy(self) -> bool:
        return bool(self._active_plan)

    @property
    def settled_objects(self) -> set[str]:
        return set(self._settled_objects)

    def execute(self, decision: Decision) -> ExecutionStatus:
        if self.busy:
            return ExecutionStatus(False, False, "Executor is already moving.")
        if decision.action not in {"pick_and_place", "sort_all_matching"}:
            return ExecutionStatus(False, False, f"No executable action: {decision.action}")
        if decision.target_object not in self.object_ids:
            return ExecutionStatus(False, False, f"Unknown target object: {decision.target_object}")
        if decision.target_object in self._settled_objects:
            return ExecutionStatus(False, True, f"{decision.target_object} already sorted.")

        obj_pos = self.object_position(decision.target_object)
        dest = self.bin_positions.get(decision.destination, self.bin_positions["reject_bin"])
        table_z = max(float(obj_pos[2]), 0.90)
        lift_z = 0.94
        drop = np.array([dest[0], dest[1], 0.92], dtype=float)

        self._active_object = decision.target_object
        self._active_destination = decision.destination
        self._last_active_pos = obj_pos.copy()
        self._active_plan = self._interpolate_path(
            [
                obj_pos,
                np.array([obj_pos[0], obj_pos[1], lift_z], dtype=float),
                np.array([dest[0], dest[1], lift_z], dtype=float),
                np.array([dest[0], dest[1], table_z], dtype=float),
                drop,
            ],
            steps_per_segment=45,
        )
        self._step_idx = 0
        return ExecutionStatus(True, False, f"Started pick/place for {decision.target_object}.")

    def step(self) -> ExecutionStatus:
        self.pin_all_objects(exclude_active=True)
        if not self._active_plan:
            return ExecutionStatus(False, True, "Idle.")
        pos = self._active_plan.pop(0)
        self.set_gripper(pos)
        self.set_object_position(self._active_object, pos)
        self._last_active_pos = pos.copy()
        if not self._active_plan and self._active_object:
            self._settled_objects.add(self._active_object)
            self._settled_destinations[self._active_object] = self._active_destination
            done = self._active_object
            self._active_object = ""
            self._last_active_pos = None
            self.pin_all_objects()
            return ExecutionStatus(False, True, f"Completed pick/place for {done}.")
        return ExecutionStatus(False, False, "Moving.")

    def stabilize_after_physics(self) -> None:
        self.pin_all_objects(exclude_active=True)
        if self._active_object and self._last_active_pos is not None:
            self.set_object_position(self._active_object, self._last_active_pos)

    def set_gripper(self, pos: np.ndarray) -> None:
        self.data.mocap_pos[self._mocap_id] = pos
        self.data.mocap_quat[self._mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])

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

    def pin_all_objects(self, exclude_active: bool = False) -> None:
        for object_id in self.object_ids:
            if exclude_active and object_id == self._active_object:
                continue
            if object_id in self._settled_objects:
                dest_name = self._settled_destinations.get(object_id, "reject_bin")
                dest = self.bin_positions.get(dest_name, self.bin_positions["reject_bin"])
                offset = self._bin_offset(object_id)
                self.set_object_position(object_id, np.array([dest[0], dest[1], 0.92], dtype=float) + offset)
            else:
                self.set_object_position(object_id, self._home_positions[object_id])

    def _bin_drop_position(self, body_name: str) -> np.ndarray:
        body_id = self.model.body(body_name).id
        pos = self.data.xpos[body_id].copy()
        return np.array([pos[0], pos[1], 0.86], dtype=float)

    def _bin_offset(self, object_id: str) -> np.ndarray:
        offsets = {
            "red_part": np.array([0.00, 0.04, 0.0], dtype=float),
            "dark_part": np.array([0.00, -0.04, 0.0], dtype=float),
            "green_part": np.array([0.00, 0.00, 0.0], dtype=float),
        }
        return offsets.get(object_id, np.zeros(3, dtype=float))

    def _interpolate_path(
        self,
        waypoints: list[np.ndarray],
        steps_per_segment: int,
    ) -> list[np.ndarray]:
        path: list[np.ndarray] = []
        for start, end in zip(waypoints, waypoints[1:]):
            for i in range(steps_per_segment):
                t = (i + 1) / steps_per_segment
                t = t * t * (3.0 - 2.0 * t)
                pos = start * (1.0 - t) + end * t
                path.append(pos)
        return path
