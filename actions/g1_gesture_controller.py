"""Visible Unitree G1 gesture control for the MuJoCo factory demo."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import mujoco


@dataclass
class _PoseSegment:
    name: str
    target: dict[str, float]
    duration: float
    elapsed: float = 0.0
    start: dict[str, float] | None = None


class G1GestureController:
    """Animates named G1 task gestures without solving whole-body balance.

    The official G1 model exposes torque actuators. For this visual POC we set
    joint positions directly and mirror values into matching actuator controls
    when available. Gravity is disabled in the showcase scene, so this gives
    stable, readable task gestures while keeping real balance control out of
    scope.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.dt = float(model.opt.timestep)
        self.joint_qpos: dict[str, int] = {}
        self.actuators: dict[str, int] = {}
        self._queue: deque[_PoseSegment] = deque()
        self.current_pose = "home"
        self.enabled = self._map_model()
        self.poses = self._build_poses()
        if self.enabled:
            self.apply_pose("home")

    def _map_model(self) -> bool:
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                self.joint_qpos[name] = int(self.model.jnt_qposadr[i])
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                self.actuators[name] = i
        return "right_shoulder_pitch_joint" in self.joint_qpos

    def _build_poses(self) -> dict[str, dict[str, float]]:
        base = {
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "left_shoulder_pitch_joint": 0.15,
            "left_shoulder_roll_joint": 0.18,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.35,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.15,
            "right_shoulder_roll_joint": -0.18,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.35,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        }
        poses = {"home": base}
        poses["inspect"] = base | {
            "waist_yaw_joint": -0.16,
            "waist_pitch_joint": 0.28,
            "left_shoulder_pitch_joint": 0.62,
            "left_shoulder_roll_joint": 0.36,
            "right_shoulder_pitch_joint": 0.62,
            "right_shoulder_roll_joint": -0.36,
            "right_elbow_joint": 0.72,
        }
        poses["reach_red"] = base | {
            "waist_yaw_joint": 0.35,
            "waist_pitch_joint": 0.18,
            "left_shoulder_pitch_joint": 0.45,
            "left_shoulder_roll_joint": 0.34,
            "right_shoulder_pitch_joint": -0.85,
            "right_shoulder_roll_joint": -0.92,
            "right_shoulder_yaw_joint": 0.48,
            "right_elbow_joint": 1.28,
            "right_wrist_pitch_joint": -0.38,
        }
        poses["reach_dark"] = base | {
            "waist_yaw_joint": -0.26,
            "waist_pitch_joint": 0.18,
            "left_shoulder_pitch_joint": 0.42,
            "left_shoulder_roll_joint": 0.30,
            "right_shoulder_pitch_joint": -0.78,
            "right_shoulder_roll_joint": -0.82,
            "right_shoulder_yaw_joint": -0.42,
            "right_elbow_joint": 1.22,
            "right_wrist_pitch_joint": -0.34,
        }
        poses["reach_green"] = base | {
            "waist_yaw_joint": 0.05,
            "waist_pitch_joint": 0.16,
            "left_shoulder_pitch_joint": 0.42,
            "right_shoulder_pitch_joint": -0.82,
            "right_shoulder_roll_joint": -0.86,
            "right_shoulder_yaw_joint": 0.05,
            "right_elbow_joint": 1.24,
            "right_wrist_pitch_joint": -0.36,
        }
        poses["reject"] = base | {
            "waist_yaw_joint": 0.62,
            "waist_pitch_joint": 0.10,
            "left_shoulder_pitch_joint": 0.38,
            "left_shoulder_roll_joint": 0.26,
            "right_shoulder_pitch_joint": -0.42,
            "right_shoulder_roll_joint": -1.05,
            "right_shoulder_yaw_joint": 0.72,
            "right_elbow_joint": 0.92,
            "right_wrist_yaw_joint": 0.55,
        }
        poses["good_bin"] = base | {
            "waist_yaw_joint": -0.52,
            "waist_pitch_joint": 0.10,
            "right_shoulder_pitch_joint": -0.38,
            "right_shoulder_roll_joint": -0.98,
            "right_shoulder_yaw_joint": -0.58,
            "right_elbow_joint": 0.90,
            "right_wrist_yaw_joint": -0.45,
        }
        poses["celebrate_done"] = base | {
            "left_shoulder_pitch_joint": -0.45,
            "left_shoulder_roll_joint": 0.65,
            "left_elbow_joint": 0.55,
            "right_shoulder_pitch_joint": -0.45,
            "right_shoulder_roll_joint": -0.65,
            "right_elbow_joint": 0.55,
        }
        return poses

    def start_pose(self, pose_name: str, duration: float = 1.0) -> None:
        if not self.enabled or pose_name not in self.poses:
            return
        self._queue.clear()
        self._queue.append(_PoseSegment(pose_name, self.poses[pose_name], duration))

    @property
    def busy(self) -> bool:
        return bool(self._queue)

    def start_task_sequence(self, target_object: str, destination: str = "reject_bin") -> None:
        if not self.enabled:
            return
        if target_object == "red_part":
            reach = "reach_red"
        elif target_object == "dark_part":
            reach = "reach_dark"
        elif target_object == "green_part":
            reach = "reach_green"
        else:
            reach = "inspect"
        place = "good_bin" if destination == "good_bin" else "reject"
        self._queue.clear()
        for name, duration in [
            ("inspect", 0.70),
            (reach, 1.45),
            (place, 1.35),
            ("home", 0.95),
        ]:
            self._queue.append(_PoseSegment(name, self.poses[name], duration))

    def home(self) -> None:
        self.start_pose("home", 0.8)

    def celebrate(self) -> None:
        self.start_pose("celebrate_done", 1.2)

    def step(self) -> str:
        if not self.enabled:
            return "disabled"
        if not self._queue:
            return self.current_pose
        segment = self._queue[0]
        if segment.start is None:
            segment.start = self._current_values(segment.target)
        segment.elapsed += self.dt
        alpha = min(1.0, segment.elapsed / max(segment.duration, self.dt))
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        pose = {
            joint: segment.start[joint] * (1.0 - alpha) + target * alpha
            for joint, target in segment.target.items()
        }
        self._apply_joint_values(pose)
        self.current_pose = segment.name
        if segment.elapsed >= segment.duration:
            self._queue.popleft()
        return self.current_pose

    def apply_pose(self, pose_name: str) -> None:
        if pose_name in self.poses:
            self._apply_joint_values(self.poses[pose_name])
            self.current_pose = pose_name

    def _current_values(self, target: dict[str, float]) -> dict[str, float]:
        return {
            joint: float(self.data.qpos[qpos_adr])
            for joint, qpos_adr in self.joint_qpos.items()
            if joint in target
        }

    def _apply_joint_values(self, values: dict[str, float]) -> None:
        for joint, value in values.items():
            qpos_adr = self.joint_qpos.get(joint)
            if qpos_adr is None:
                continue
            self.data.qpos[qpos_adr] = value
            joint_id = self.model.joint(joint).id
            dof_adr = self.model.jnt_dofadr[joint_id]
            self.data.qvel[dof_adr] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
