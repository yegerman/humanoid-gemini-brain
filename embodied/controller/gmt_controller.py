"""Whole-body G1 controller backed by GMT (General Motion Tracking).

Wraps the pretrained 23-DoF GMT policy (legs + waist + arms) so a brain can drive it.
The policy TRACKS a reference motion (root height/orientation/velocity + 23 joint targets
per frame); a "skill" is therefore a synthesized motion trajectory. Runs on CPU.

Refactored from `sim2sim.py` of github.com/zixuan417/humanoid-general-motion-tracking
(Apache-2.0). Differences: headless-capable, built-in `mujoco.viewer`, `get_proprio()`,
switchable motion, and a step()/run() split.
"""
from __future__ import annotations

import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mujoco
import torch

GMT = Path(__file__).resolve().parent / "gmt"
sys.path.insert(0, str(GMT))  # so `utils.motion_lib` (with its relative imports) resolves
from utils.motion_lib import MotionLib  # noqa: E402

MODEL_PATH = GMT / "g1" / "g1.xml"
STAGE_SCENE = GMT / "g1" / "scene_stage.xml"
POLICY_PATH = GMT / "pretrained.pt"
MOTIONS = GMT / "motions"

DOF_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
]


def quat_to_euler(quat):  # quat = (w,x,y,z) from mujoco framequat sensor
    qw, qx, qy, qz = quat
    e = np.zeros(3)
    e[0] = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    sinp = 2 * (qw * qy - qz * qx)
    e[1] = np.copysign(np.pi / 2, sinp) if abs(sinp) >= 1 else np.arcsin(sinp)
    e[2] = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return e


@dataclass
class Proprio:
    pos: np.ndarray
    quat: np.ndarray      # (w,x,y,z)
    yaw: float
    height: float
    upright: bool
    dof_pos: np.ndarray   # 23 joint positions


class GMTController:
    def __init__(self, motion: str = "basic_walk.pkl", device: str = "cpu",
                 scene: str | Path | None = None) -> None:
        self.device = device
        self.num_actions = 23
        self.num_dofs = 23
        self.stiffness = np.array([100,100,100,150,40,40, 100,100,100,150,40,40,
                                   150,150,150, 40,40,40,40, 40,40,40,40], dtype=np.float32)
        self.damping = np.array([2,2,2,4,2,2, 2,2,2,4,2,2, 4,4,4, 5,5,5,5, 5,5,5,5], dtype=np.float32)
        self.default_dof_pos = np.array([-0.2,0,0,0.4,-0.2,0, -0.2,0,0,0.4,-0.2,0,
                                         0,0,0, 0,0.4,0,1.2, 0,-0.4,0,1.2], dtype=np.float32)
        self.torque_limits = np.array([88,139,88,139,50,50, 88,139,88,139,50,50,
                                       88,50,50, 25,25,25,25, 25,25,25,25], dtype=np.float32)

        self.sim_dt = 0.001
        self.sim_decimation = 20
        self.control_dt = self.sim_dt * self.sim_decimation  # 0.02 -> 50 Hz
        self.action_scale = 0.5
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        self.ang_vel_scale = 0.25
        self.tar_obs_steps = torch.tensor(
            [1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95],
            device=device, dtype=torch.int)
        self.n_proprio = 3 + 2 + 3 * self.num_actions
        self.history_len = 20

        self.model = mujoco.MjModel.from_xml_path(str(scene or MODEL_PATH))
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_step(self.model, self.data)

        self.policy = torch.jit.load(str(POLICY_PATH), map_location=device)
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.pd_target = self.default_dof_pos.copy()
        self.counter = 0
        self.steer_yaw_rate = 0.0   # rad/s injected into mimic to steer heading while walking
        self.fwd_scale = 1.0        # scale forward reference velocity (0 => walk in place)
        self.proprio_history = deque(maxlen=self.history_len)
        for _ in range(self.history_len):
            self.proprio_history.append(np.zeros(self.n_proprio))

        self.set_motion(motion)

    # --- brain-facing API -------------------------------------------------
    def set_motion(self, motion: str, reset_time: bool = True, force: bool = False) -> None:
        """Switch the reference motion the policy tracks (file name or absolute path)."""
        path = motion if Path(motion).is_absolute() else str(MOTIONS / motion)
        if not force and getattr(self, "_motion_path", None) == path:
            return  # already tracking this motion; avoid redundant reload
        self._motion_path = path
        self._motion_lib = MotionLib(path, self.device)
        if reset_time:
            self.motion_t0 = self.counter

    def get_proprio(self) -> Proprio:
        d = self.data
        quat = d.qpos[3:7].copy()  # (w,x,y,z)
        e = quat_to_euler(quat)
        gz_upright = abs(e[0]) < 0.7 and abs(e[1]) < 0.7
        return Proprio(pos=d.qpos[0:3].copy(), quat=quat, yaw=float(e[2]),
                       height=float(d.qpos[2]), upright=bool(gz_upright and d.qpos[2] > 0.5),
                       dof_pos=d.qpos[-self.num_dofs:].copy())

    # --- simulation -------------------------------------------------------
    def step(self) -> None:
        d = self.data
        dof_pos = d.qpos.astype(np.float32)[-self.num_dofs:]
        dof_vel = d.qvel.astype(np.float32)[-self.num_dofs:]
        if self.counter % self.sim_decimation == 0:
            ct = (self.counter - getattr(self, "motion_t0", 0)) // self.sim_decimation
            mimic = self._get_mimic_obs(max(ct, 0))
            quat = d.sensor('orientation').data.astype(np.float32)  # (w,x,y,z)? -> framequat is (w,x,y,z)
            ang_vel = d.sensor('angular-velocity').data.astype(np.float32)
            rpy = quat_to_euler(quat)
            odv = dof_vel.copy(); odv[[4, 5, 10, 11]] = 0.0
            obs_prop = np.concatenate([
                ang_vel * self.ang_vel_scale, rpy[:2],
                (dof_pos - self.default_dof_pos) * self.dof_pos_scale,
                odv * self.dof_vel_scale, self.last_action])
            obs_hist = np.array(self.proprio_history).flatten()
            obs = np.concatenate([mimic, obs_prop, obs_hist])
            with torch.no_grad():
                raw = self.policy(torch.from_numpy(obs).float().unsqueeze(0)).cpu().numpy().squeeze()
            self.last_action = raw.copy()
            self.pd_target = np.clip(raw, -10, 10) * self.action_scale + self.default_dof_pos
            self.proprio_history.append(obs_prop)
        torque = (self.pd_target - dof_pos) * self.stiffness - dof_vel * self.damping
        torque = np.clip(torque, -self.torque_limits, self.torque_limits)
        if not np.all(np.isfinite(torque)):
            self._recover(); return
        d.ctrl = torque
        mujoco.mj_step(self.model, self.data)
        self.counter += 1
        # Guard: if the policy diverged (NaN/Inf in state), recover instead of letting
        # NaN propagate into the renderer and segfault the whole process.
        if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
            self._recover()

    def recover_to_stand(self, verbose: bool = True) -> None:
        """Reset to the safe standing keyframe and restart the current reference clip from a
        matched neutral pose. Used both as the NaN-instability guard and as the user-invokable
        'stand up' / recover skill (GMT has no animated floor get-up clip — this is a clean
        state reset to a balanced standing pose)."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self.last_action[:] = 0.0
        self.pd_target = self.default_dof_pos.copy()
        self.proprio_history.clear()
        for _ in range(self.history_len):
            self.proprio_history.append(np.zeros(self.n_proprio))
        self.steer_yaw_rate = 0.0
        self.fwd_scale = 0.0
        self.motion_t0 = self.counter   # restart the clip from this matched standing pose
        if verbose:
            print("  [controller] recovered to a stable standing pose.")

    def _recover(self) -> None:
        """Physics went unstable (NaN/Inf QACC) — recover so a bad motion transition
        stumbles-and-recovers rather than crashing the native renderer."""
        self.recover_to_stand(verbose=False)
        print("  [controller] instability detected -> recovered to standing.")

    def _get_mimic_obs(self, curr_time_step: int) -> np.ndarray:
        ts = self.tar_obs_steps
        n = len(ts)
        motion_times = torch.tensor([curr_time_step * self.control_dt], device=self.device).unsqueeze(-1)
        obs_times = (ts * self.control_dt + motion_times).flatten()
        motion_ids = torch.zeros(n, dtype=torch.int, device=self.device)
        rp, rr, rv, rav, dp, _ = self._motion_lib.calc_motion_frame(motion_ids, obs_times)
        # rr is (n,4) quat in (x,y,z,w)
        x, y, z, w = rr[:, 0], rr[:, 1], rr[:, 2], rr[:, 3]
        roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)).reshape(1, n, 1)
        pitch = torch.asin(torch.clip(2 * (w * y - z * x), -1, 1)).reshape(1, n, 1)
        rv = self._rot_inv(rr, rv).reshape(1, n, 3)
        rav = self._rot_inv(rr, rav).reshape(1, n, 3)
        rp = rp.reshape(1, n, 3)
        dp = dp.reshape(1, n, -1)
        rav_z = rav[..., 2:3]
        # Steering: override the reference yaw-rate (and optionally forward speed) so the
        # robot tracks a commanded heading change while keeping the clip's walking gait.
        if self.steer_yaw_rate != 0.0:
            rav_z = torch.full_like(rav_z, float(self.steer_yaw_rate))
        if self.fwd_scale != 1.0:
            rv = rv * float(self.fwd_scale)
        mimic = torch.cat((rp[..., 2:3], roll, pitch, rv, rav_z, dp), dim=-1)
        return mimic.reshape(1, -1).detach().cpu().numpy().squeeze()

    @staticmethod
    def _rot_inv(q, v):  # quat_rotate_inverse, q=(x,y,z,w)
        qw = q[:, -1]
        qvec = q[:, :3]
        a = v * (2.0 * qw ** 2 - 1.0).unsqueeze(-1)
        b = torch.cross(qvec, v, dim=-1) * qw.unsqueeze(-1) * 2.0
        c = qvec * torch.bmm(qvec.view(-1, 1, 3), v.view(-1, 3, 1)).squeeze(-1) * 2.0
        return a - b + c

    def run(self, duration: float = 60.0, view: bool = True, realtime: bool = True) -> None:
        if view:
            import mujoco.viewer
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                start = time.time()
                while viewer.is_running() and time.time() - start < duration:
                    t0 = time.time()
                    self.step()
                    if self.counter % 4 == 0:
                        viewer.sync()
                    if realtime:
                        dt = self.sim_dt - (time.time() - t0)
                        if dt > 0:
                            time.sleep(dt)
        else:
            for _ in range(int(duration / self.sim_dt)):
                self.step()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", default="basic_walk.pkl")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--no-view", action="store_true")
    a = ap.parse_args()
    c = GMTController(motion=a.motion)
    c.run(duration=a.duration, view=not a.no_view)
