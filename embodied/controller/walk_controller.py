"""Low-level G1 walking controller.

Forked from unitree_rl_gym `deploy/deploy_mujoco/deploy_mujoco.py`. The observation +
PD-control + policy math is preserved verbatim; behavioural changes:
  * the velocity command `(vx, vy, vyaw)` is **live** (`set_cmd_vel`) not read-once,
  * `get_proprio()` exposes feedback,
  * a `step()`/`run()` split so a brain can drive it,
  * optional **full-body model**: the 12-DoF legs policy drives the legs of `g1_29dof`
    while the waist + arms (17 extra actuators) are PD-held at a pose the skill library
    can override (`set_hold_pose`) — this is how arm gestures work while balancing.

The pretrained policy is the 12-DoF legs-only walker: obs=47, num_actions=12, ~50 Hz.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

ASSETS = Path(__file__).resolve().parent / "assets"
SCENE_12DOF = ASSETS / "g1_description" / "scene.xml"
SCENE_29DOF_STAGE = ASSETS / "g1_description" / "scene_stage_29dof.xml"

# Balanced standing pose for the 17 non-leg joints of g1_29dof (joints 13..29),
# in actuator order: waist(3) + left arm(7) + right arm(7).
HOLD_POSE_29 = {
    "waist_yaw_joint": 0.0, "waist_roll_joint": 0.0, "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.20, "left_shoulder_roll_joint": 0.20,
    "left_shoulder_yaw_joint": 0.0, "left_elbow_joint": 0.50,
    "left_wrist_roll_joint": 0.0, "left_wrist_pitch_joint": 0.0, "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.20, "right_shoulder_roll_joint": -0.20,
    "right_shoulder_yaw_joint": 0.0, "right_elbow_joint": 0.50,
    "right_wrist_roll_joint": 0.0, "right_wrist_pitch_joint": 0.0, "right_wrist_yaw_joint": 0.0,
}


def get_gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    g = np.zeros(3)
    g[0] = 2 * (-qz * qx + qw * qy)
    g[1] = -2 * (qz * qy + qw * qx)
    g[2] = 1 - 2 * (qw * qw + qz * qz)
    return g


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


@dataclass
class Proprio:
    pos: np.ndarray          # base xyz (m)
    quat: np.ndarray         # base orientation (w,x,y,z)
    lin_vel: np.ndarray      # base linear velocity (m/s)
    ang_vel: np.ndarray      # base angular velocity (rad/s)
    yaw: float               # base heading (rad)
    height: float            # base height (m)
    upright: bool            # torso still upright (not fallen)
    joint_pos: np.ndarray    # all actuated joint positions


class WalkController:
    def __init__(self, config_path=None, xml_path=None, full_body=False) -> None:
        config_path = Path(config_path) if config_path else ASSETS / "g1.yaml"
        with open(config_path, "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        self.full_body = full_body
        if xml_path is None:
            xml_path = SCENE_29DOF_STAGE if full_body else SCENE_12DOF
        policy_path = ASSETS / "motion.pt"

        self.simulation_dt = cfg["simulation_dt"]
        self.control_decimation = cfg["control_decimation"]
        self.kps = np.array(cfg["kps"], dtype=np.float32)          # 12 leg gains
        self.kds = np.array(cfg["kds"], dtype=np.float32)
        self.default_angles = np.array(cfg["default_angles"], dtype=np.float32)
        self.ang_vel_scale = cfg["ang_vel_scale"]
        self.dof_pos_scale = cfg["dof_pos_scale"]
        self.dof_vel_scale = cfg["dof_vel_scale"]
        self.action_scale = cfg["action_scale"]
        self.cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float32)
        self.num_actions = cfg["num_actions"]                       # 12
        self.num_obs = cfg["num_obs"]                               # 47

        self.cmd = np.array(cfg["cmd_init"], dtype=np.float32)
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.counter = 0

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.simulation_dt
        self.policy = torch.jit.load(str(policy_path))

        n = self.num_actions
        # qpos layout: [0:7] free base, [7:7+12] legs, [19:] extra (waist+arms)
        self.leg_q = slice(7, 7 + n)
        self.leg_dq = slice(6, 6 + n)
        self.n_extra = self.model.nu - n
        self._name2jid = {
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i): i
            for i in range(self.model.njnt)
        }

        if self.full_body and self.n_extra > 0:
            # extra actuators are indices [12:nu] in joint order 13..29
            self.extra_names = [
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n + i)
                for i in range(self.n_extra)
            ]
            # The leg policy can't see the waist/arms, so they must be STIFF — emulating the
            # rigid lumped torso the 12-DoF policy was trained on. Waist stiffest of all.
            self.extra_kps = np.array(
                [400.0 if "waist" in nm else 120.0 for nm in self.extra_names], dtype=np.float32)
            self.extra_kds = np.array(
                [10.0 if "waist" in nm else 4.0 for nm in self.extra_names], dtype=np.float32)
            self.hold_targets = np.array(
                [HOLD_POSE_29.get(nm, 0.0) for nm in self.extra_names], dtype=np.float32
            )
            self._init_standing()
        else:
            self.extra_names = []
            self.hold_targets = np.zeros(0, dtype=np.float32)

    def _init_standing(self) -> None:
        """Place the full-body model in a standing pose at correct height."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.793
        self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qpos[self.leg_q] = self.default_angles
        self.data.qpos[7 + self.num_actions:] = self.hold_targets
        mujoco.mj_forward(self.model, self.data)

    # --- brain-facing API -------------------------------------------------
    def set_cmd_vel(self, vx: float, vy: float = 0.0, vyaw: float = 0.0) -> None:
        self.cmd[:] = (float(vx), float(vy), float(vyaw))

    def set_hold_pose(self, pose: dict[str, float]) -> None:
        """Override held waist/arm targets by joint name (used by the skill library)."""
        for i, nm in enumerate(self.extra_names):
            if nm in pose:
                self.hold_targets[i] = float(pose[nm])

    def get_proprio(self) -> Proprio:
        d = self.data
        quat = d.qpos[3:7].copy()
        g = get_gravity_orientation(quat)
        w, x, y, z = quat
        yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        return Proprio(
            pos=d.qpos[0:3].copy(), quat=quat,
            lin_vel=d.qvel[0:3].copy(), ang_vel=d.qvel[3:6].copy(),
            yaw=yaw, height=float(d.qpos[2]),
            upright=bool(g[2] < -0.6), joint_pos=d.qpos[7:].copy(),
        )

    # --- simulation loop --------------------------------------------------
    def _update_policy(self) -> None:
        d = self.data
        qj = (d.qpos[self.leg_q] - self.default_angles) * self.dof_pos_scale
        dqj = d.qvel[self.leg_dq] * self.dof_vel_scale
        omega = d.qvel[3:6] * self.ang_vel_scale
        gravity = get_gravity_orientation(d.qpos[3:7])

        period = 0.8
        count = self.counter * self.simulation_dt
        phase = count % period / period
        n = self.num_actions
        self.obs[:3] = omega
        self.obs[3:6] = gravity
        self.obs[6:9] = self.cmd * self.cmd_scale
        self.obs[9:9 + n] = qj
        self.obs[9 + n:9 + 2 * n] = dqj
        self.obs[9 + 2 * n:9 + 3 * n] = self.action
        self.obs[9 + 3 * n:9 + 3 * n + 2] = (np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase))

        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        self.action = self.policy(obs_tensor).detach().numpy().squeeze()
        self.target_dof_pos = self.action * self.action_scale + self.default_angles

    def step(self) -> None:
        d = self.data
        n = self.num_actions
        leg_tau = pd_control(self.target_dof_pos, d.qpos[self.leg_q], self.kps,
                             np.zeros(n, dtype=np.float32), d.qvel[self.leg_dq], self.kds)
        d.ctrl[:n] = leg_tau
        if self.n_extra > 0:
            q_extra = d.qpos[7 + n:]
            dq_extra = d.qvel[6 + n:]
            extra_tau = pd_control(self.hold_targets, q_extra, self.extra_kps,
                                   np.zeros(self.n_extra, dtype=np.float32), dq_extra, self.extra_kds)
            d.ctrl[n:] = extra_tau
        mujoco.mj_step(self.model, self.data)
        self.counter += 1
        if self.counter % self.control_decimation == 0:
            self._update_policy()

    def run(self, duration: float = 60.0, view: bool = True, realtime: bool = True) -> None:
        if view:
            import mujoco.viewer
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                start = time.time()
                while viewer.is_running() and time.time() - start < duration:
                    t0 = time.time()
                    self.step()
                    viewer.sync()
                    if realtime:
                        dt = self.model.opt.timestep - (time.time() - t0)
                        if dt > 0:
                            time.sleep(dt)
        else:
            for _ in range(int(duration / self.simulation_dt)):
                self.step()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--vx", type=float, default=0.5)
    ap.add_argument("--full-body", action="store_true")
    ap.add_argument("--no-view", action="store_true")
    a = ap.parse_args()
    c = WalkController(full_body=a.full_body)
    c.set_cmd_vel(a.vx, 0.0, 0.0)
    c.run(duration=a.duration, view=not a.no_view)
