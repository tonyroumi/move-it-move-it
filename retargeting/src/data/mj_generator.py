from __future__ import annotations

import numpy as np
import mujoco
from dataclasses import dataclass
from scipy.interpolate import CubicSpline


# =============================================================================
# Skeleton Analysis
# =============================================================================

@dataclass
class JointInfo:
    id: int
    name: str
    type: int
    body_id: int
    qpos_adr: int
    dof_adr: int
    n_qpos: int
    n_dof: int
    range: np.ndarray | None
    axis: np.ndarray


@dataclass
class BodyInfo:
    id: int
    name: str
    parent_id: int
    children: list
    joints: list
    depth: int = 0
    is_leaf: bool = False


class SkeletonAnalyzer:
    """Extracts kinematic tree structure from any MuJoCo model."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.bodies: dict[int, BodyInfo] = {}
        self.joints: dict[int, JointInfo] = {}
        self.kinematic_chains: list[list[int]] = []
        self.adjacency: dict[int, list[int]] = {}
        self.actuator_to_joint: dict[int, int] = {}
        self._analyze()

    def _analyze(self):
        model = self.model

        for jid in range(model.njnt):
            jnt_type = model.jnt_type[jid]
            if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                n_qpos, n_dof = 7, 6
            elif jnt_type == mujoco.mjtJoint.mjJNT_BALL:
                n_qpos, n_dof = 4, 3
            else:
                n_qpos, n_dof = 1, 1

            has_limit = bool(model.jnt_limited[jid])
            jnt_range = model.jnt_range[jid].copy() if has_limit else None

            self.joints[jid] = JointInfo(
                id=jid, name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"joint_{jid}",
                type=jnt_type, body_id=model.jnt_bodyid[jid],
                qpos_adr=model.jnt_qposadr[jid], dof_adr=model.jnt_dofadr[jid],
                n_qpos=n_qpos, n_dof=n_dof, range=jnt_range, axis=model.jnt_axis[jid].copy(),
            )

        for bid in range(model.nbody):
            body_joints = [j for j in self.joints.values() if j.body_id == bid]
            self.bodies[bid] = BodyInfo(
                id=bid, name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body_{bid}",
                parent_id=model.body_parentid[bid], children=[], joints=body_joints,
            )

        for bid, body in self.bodies.items():
            if bid == 0:
                continue
            self.bodies[body.parent_id].children.append(bid)

        self._compute_depths(0, 0)
        for bid, body in self.bodies.items():
            body.is_leaf = len(body.children) == 0 and bid != 0

        for bid, body in self.bodies.items():
            neighbors = list(body.children)
            if bid != 0:
                neighbors.append(body.parent_id)
            self.adjacency[bid] = neighbors

        for bid, body in self.bodies.items():
            if body.is_leaf:
                self.kinematic_chains.append(self._trace_to_root(bid))

        for aid in range(model.nu):
            if model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT:
                self.actuator_to_joint[aid] = model.actuator_trnid[aid, 0]

    def _compute_depths(self, bid, depth):
        self.bodies[bid].depth = depth
        for cid in self.bodies[bid].children:
            self._compute_depths(cid, depth + 1)

    def _trace_to_root(self, leaf_id):
        chain = [leaf_id]
        current = leaf_id
        while current != 0:
            current = self.bodies[current].parent_id
            chain.append(current)
        chain.reverse()
        return chain

    def detect_end_effectors(self) -> list[int]:
        return [bid for bid, body in self.bodies.items() if body.is_leaf]

    def get_joints_in_chain(self, chain: list[int]) -> list[JointInfo]:
        joints = []
        for bid in chain:
            joints.extend(self.bodies[bid].joints)
        return joints

    def get_all_joint_ranges(self) -> tuple[np.ndarray, np.ndarray]:
        model = self.model
        lower = np.full(model.nq, -np.pi)
        upper = np.full(model.nq, np.pi)
        for jinfo in self.joints.values():
            adr = jinfo.qpos_adr
            if jinfo.type == mujoco.mjtJoint.mjJNT_FREE:
                lower[adr:adr+3] = -100.0; upper[adr:adr+3] = 100.0
                lower[adr+3:adr+7] = -1.0; upper[adr+3:adr+7] = 1.0
            elif jinfo.type == mujoco.mjtJoint.mjJNT_BALL:
                lower[adr:adr+4] = -1.0; upper[adr:adr+4] = 1.0
            else:
                if jinfo.range is not None:
                    lower[adr] = jinfo.range[0]; upper[adr] = jinfo.range[1]
                elif jinfo.type == mujoco.mjtJoint.mjJNT_SLIDE:
                    lower[adr] = -2.0; upper[adr] = 2.0
        return lower, upper

    def get_com_position(self, data: mujoco.MjData) -> np.ndarray:
        return data.subtree_com[0].copy()


# =============================================================================
# Control Utilities
# =============================================================================

class ControlFilter:
    """
    First-order low-pass filter on control signals.
    Eliminates high-frequency jitter from any control source.

        y[n] = alpha * x[n] + (1 - alpha) * y[n-1]

    Alpha is computed from a cutoff frequency and the simulation timestep:
        alpha = dt / (dt + 1/(2*pi*fc))
    """

    def __init__(self, size: int, dt: float, cutoff_hz: float = 5.0):
        tau = 1.0 / (2.0 * np.pi * cutoff_hz)
        self.alpha = dt / (dt + tau)
        self.y = np.zeros(size)
        self.initialized = False

    def filter(self, x: np.ndarray) -> np.ndarray:
        if not self.initialized:
            self.y[:] = x
            self.initialized = True
            return self.y.copy()
        self.y += self.alpha * (x - self.y)
        return self.y.copy()

    def reset(self, x: np.ndarray | None = None):
        if x is not None:
            self.y[:] = x
        else:
            self.y[:] = 0.0
        self.initialized = x is not None


class PDController:
    """
    PD controller with gravity compensation.
    Detects position-servo vs torque actuators automatically.
    """

    def __init__(self, model: mujoco.MjModel, kp_scale=1.0, kd_ratio=0.5,
                 max_pos_error=0.5, max_vel=10.0):
        self.model = model
        self.nu = model.nu
        self.kp = np.zeros(model.nu)
        self.kd = np.zeros(model.nu)
        self.ctrl_range = np.zeros((model.nu, 2))
        self.is_position_servo = np.zeros(model.nu, dtype=bool)
        self.max_pos_error = max_pos_error
        self.max_vel = max_vel
        self._configure(kp_scale, kd_ratio)

    def _configure(self, kp_scale, kd_ratio):
        model = self.model
        for aid in range(model.nu):
            gaintype = model.actuator_gaintype[aid]
            biastype = model.actuator_biastype[aid]

            if model.actuator_ctrllimited[aid]:
                self.ctrl_range[aid] = model.actuator_ctrlrange[aid]
            elif model.actuator_forcelimited[aid]:
                self.ctrl_range[aid] = model.actuator_forcerange[aid]
            else:
                self.ctrl_range[aid] = [-100.0, 100.0]

            if (biastype == mujoco.mjtBias.mjBIAS_AFFINE and
                gaintype == mujoco.mjtGain.mjGAIN_AFFINE and
                abs(model.actuator_gainprm[aid, 0]) > 0):
                self.is_position_servo[aid] = True
                self.kp[aid] = kp_scale
                continue

            force_limit = max(abs(self.ctrl_range[aid, 0]), abs(self.ctrl_range[aid, 1]), 1.0)
            if gaintype == mujoco.mjtGain.mjGAIN_FIXED:
                gain = max(abs(model.actuator_gainprm[aid, 0]), 1e-6)
                self.kp[aid] = (force_limit * 0.3 / (gain * self.max_pos_error)) * kp_scale
                self.kd[aid] = 2.0 * np.sqrt(self.kp[aid]) * kd_ratio
            else:
                self.kp[aid] = 15.0 * kp_scale
                self.kd[aid] = 2.0 * np.sqrt(15.0) * kd_ratio

    def compute(self, data, q_target):
        model = self.model
        ctrl = np.zeros(model.nu)

        for aid in range(model.nu):
            if model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT:
                continue
            jnt_id = model.actuator_trnid[aid, 0]
            jnt_type = model.jnt_type[jnt_id]
            qpos_adr = model.jnt_qposadr[jnt_id]
            dof_adr = model.jnt_dofadr[jnt_id]

            if jnt_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                q_err = np.clip(q_target[qpos_adr] - data.qpos[qpos_adr],
                                -self.max_pos_error, self.max_pos_error)
                if self.is_position_servo[aid]:
                    ctrl[aid] = data.qpos[qpos_adr] + q_err * self.kp[aid]
                else:
                    qd = np.clip(-data.qvel[dof_adr], -self.max_vel, self.max_vel)
                    ctrl[aid] = self.kp[aid] * q_err + self.kd[aid] * qd
                    ctrl[aid] += data.qfrc_bias[dof_adr]

            elif jnt_type == mujoco.mjtJoint.mjJNT_BALL:
                q_curr = data.qpos[qpos_adr:qpos_adr+4]
                q_targ = q_target[qpos_adr:qpos_adr+4]
                nc = np.linalg.norm(q_curr)
                nt = np.linalg.norm(q_targ)
                if nc > 1e-10: q_curr = q_curr / nc
                if nt > 1e-10: q_targ = q_targ / nt
                q_err = _quat_mul(_quat_conj(q_curr), q_targ)
                if q_err[0] < 0: q_err = -q_err
                aa = 2.0 * q_err[1:4]
                mag = np.linalg.norm(aa)
                if mag > self.max_pos_error: aa *= self.max_pos_error / mag
                qd = np.clip(-data.qvel[dof_adr:dof_adr+3], -self.max_vel, self.max_vel)
                ctrl[aid] = self.kp[aid] * aa[0] + self.kd[aid] * qd[0]
                if not self.is_position_servo[aid]:
                    ctrl[aid] += data.qfrc_bias[dof_adr]

        return np.clip(ctrl, self.ctrl_range[:, 0], self.ctrl_range[:, 1])


class OperationalSpaceController:
    """
    Computes joint torques directly from task-space position errors
    using the body Jacobian at every timestep. No IK needed.

        tau = J^T * F_task + tau_gravity

    where F_task = kp * (x_des - x) - kd * v

    This is smooth by construction since the Jacobian and EE positions
    vary continuously with the configuration.
    """

    def __init__(self, model: mujoco.MjModel, kp: float = 100.0,
                 kd: float = 20.0, max_force: float = 50.0):
        self.model = model
        self.kp = kp
        self.kd = kd
        self.max_force = max_force
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))

    def compute(self, data: mujoco.MjData,
                targets: dict[int, np.ndarray]) -> np.ndarray:
        """
        Compute joint-space torques from task-space targets.

        Args:
            data: Current simulation state
            targets: {body_id: desired_position (3,)}

        Returns:
            tau: Joint-space torques (nv,)
        """
        model = self.model
        tau = np.zeros(model.nv)

        for bid, x_des in targets.items():
            x_cur = data.xpos[bid]
            err = x_des - x_cur

            # Compute EE velocity via Jacobian
            mujoco.mj_jacBody(model, data, self.jacp, self.jacr, bid)
            v_ee = self.jacp @ data.qvel  # (3,)

            # Task-space PD force
            f_task = self.kp * err - self.kd * v_ee

            # Clamp force magnitude
            f_mag = np.linalg.norm(f_task)
            if f_mag > self.max_force:
                f_task *= self.max_force / f_mag

            # Map to joint torques: tau += J^T * f
            tau += self.jacp.T @ f_task

        # Add gravity compensation
        tau += data.qfrc_bias

        return tau

    def to_ctrl(self, model: mujoco.MjModel, tau: np.ndarray) -> np.ndarray:
        """
        Map joint-space torques to actuator control signals.
        Handles both torque actuators and position servos.
        """
        ctrl = np.zeros(model.nu)

        for aid in range(model.nu):
            if model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT:
                continue
            jnt_id = model.actuator_trnid[aid, 0]
            dof_adr = model.jnt_dofadr[jnt_id]

            gaintype = model.actuator_gaintype[aid]
            biastype = model.actuator_biastype[aid]

            if (biastype == mujoco.mjtBias.mjBIAS_AFFINE and
                gaintype == mujoco.mjtGain.mjGAIN_AFFINE):
                # Position servo: can't directly apply torque.
                # Convert desired torque to a position offset:
                # tau ≈ kp * (q_des - q) => q_des ≈ q + tau / kp
                builtin_kp = abs(model.actuator_gainprm[aid, 0])
                if builtin_kp > 1e-6:
                    qpos_adr = model.jnt_qposadr[jnt_id]
                    from_data = model  # just need qpos
                    # We'll read qpos from the caller's data — handled in task
                    ctrl[aid] = tau[dof_adr] / builtin_kp
                else:
                    ctrl[aid] = 0.0
            else:
                # Torque actuator: direct mapping
                gain = model.actuator_gainprm[aid, 0]
                if abs(gain) > 1e-6:
                    ctrl[aid] = tau[dof_adr] / gain
                else:
                    ctrl[aid] = tau[dof_adr]

            # Clamp
            if model.actuator_ctrllimited[aid]:
                lo, hi = model.actuator_ctrlrange[aid]
                ctrl[aid] = np.clip(ctrl[aid], lo, hi)

        return ctrl


# =============================================================================
# Trajectory Generation
# =============================================================================

class TrajectoryGenerator:
    def random_waypoint_spline(self, start, n_waypoints=4, duration=4.0,
                               workspace_radius=0.5, dt=0.002):
        times_wp = np.linspace(0, duration, n_waypoints + 2)
        waypoints = np.zeros((n_waypoints + 2, 3))
        waypoints[0] = start
        for i in range(1, n_waypoints + 1):
            offset = np.random.uniform(-workspace_radius, workspace_radius, size=3)
            offset[2] = abs(offset[2]) * 0.5
            waypoints[i] = start + offset
        waypoints[-1] = start + np.random.uniform(-0.05, 0.05, size=3)
        cs = CubicSpline(times_wp, waypoints, bc_type='clamped')
        times = np.arange(0, duration, dt)
        return times, cs(times)

    def sinusoidal(self, center, amplitude=0.3, frequency=0.5, phase=0.0,
                   duration=4.0, dt=0.002):
        if isinstance(amplitude, (int, float)):
            amplitude = np.array([amplitude, amplitude, amplitude * 0.5])
        if isinstance(frequency, (int, float)):
            frequency = np.array([frequency, frequency * 1.5, frequency * 0.7])
        if isinstance(phase, (int, float)):
            phase = np.array([phase, phase + np.pi/3, phase + np.pi/6])
        times = np.arange(0, duration, dt)
        positions = np.zeros((len(times), 3))
        for d in range(3):
            positions[:, d] = center[d] + amplitude[d] * np.sin(
                2 * np.pi * frequency[d] * times + phase[d]
            )
        return times, positions

    def minimum_jerk(self, start, end, duration=2.0, dt=0.002):
        times = np.arange(0, duration, dt)
        tau = times / duration
        s = 10*tau**3 - 15*tau**4 + 6*tau**5
        return times, start[None,:] + s[:,None] * (end - start)[None,:]

    def circular(self, center, radius=0.3, plane="xz", frequency=0.5,
                 duration=4.0, dt=0.002):
        times = np.arange(0, duration, dt)
        theta = 2 * np.pi * frequency * times
        positions = np.tile(center, (len(times), 1))
        ax = {"xz": (0, 2), "xy": (0, 1), "yz": (1, 2)}[plane]
        positions[:, ax[0]] += radius * np.cos(theta)
        positions[:, ax[1]] += radius * np.sin(theta)
        return times, positions

    def reach_targets(self, center, n_targets=5, radius=0.4,
                      move_time=0.8, dt=0.002):
        targets = [center.copy()]
        for _ in range(n_targets):
            off = np.random.randn(3) * radius
            off[2] = abs(off[2]) * 0.3 + 0.1
            targets.append(center + off)
        targets.append(center.copy())
        all_t, all_p = [], []
        t_off = 0.0
        for i in range(len(targets) - 1):
            t, p = self.minimum_jerk(targets[i], targets[i+1], move_time, dt)
            all_t.append(t + t_off); all_p.append(p); t_off += move_time
        return np.concatenate(all_t), np.concatenate(all_p)

    def correlated_multi(self, centers, base_freq=0.3, freq_ratios=None,
                         phase_offsets=None, amplitude=0.3, duration=4.0, dt=0.002):
        n = len(centers)
        if freq_ratios is None:
            freq_ratios = [1.0 + 0.5 * i for i in range(n)]
        if phase_offsets is None:
            phase_offsets = [np.pi * (i % 2) for i in range(n)]
        times = np.arange(0, duration, dt)
        trajs = []
        for i in range(n):
            _, t = self.sinusoidal(centers[i], amplitude, base_freq * freq_ratios[i],
                                   phase_offsets[i], duration, dt)
            trajs.append(t)
        return times, trajs

    def joint_space_spline(self, q_start, joint_ranges, n_waypoints=5,
                           duration=4.0, dt=0.002, scale=0.5):
        lower, upper = joint_ranges
        nq = len(q_start)
        times_wp = np.linspace(0, duration, n_waypoints + 2)
        wp = np.zeros((n_waypoints + 2, nq))
        wp[0] = q_start
        center = (lower + upper) / 2
        half = (upper - lower) / 2
        for i in range(1, n_waypoints + 1):
            wp[i] = center + scale * half * np.random.uniform(-1, 1, size=nq)
            wp[i] = np.clip(wp[i], lower, upper)
        wp[-1] = q_start
        cs = CubicSpline(times_wp, wp, bc_type='clamped')
        times = np.arange(0, duration, dt)
        return times, np.clip(cs(times), lower, upper)


# =============================================================================
# Tasks
# =============================================================================

class BaseTask:
    def __init__(self, model, skeleton, pd, osc, traj, ee_body_ids):
        self.model = model
        self.skeleton = skeleton
        self.pd = pd
        self.osc = osc
        self.traj = traj
        self.ee_body_ids = ee_body_ids
        self.dt = model.opt.timestep
        # Control filter: 10 Hz cutoff removes jitter while preserving motion
        self.ctrl_filter = ControlFilter(model.nu, model.opt.timestep, cutoff_hz=10.0)

    def _get_ee_positions(self, data):
        return {bid: data.xpos[bid].copy() for bid in self.ee_body_ids}

    def _filtered_ctrl(self, ctrl: np.ndarray) -> np.ndarray:
        """Apply low-pass filter and safety clamp."""
        ctrl = np.clip(ctrl, -1e3, 1e3)
        ctrl[np.isnan(ctrl)] = 0.0
        return self.ctrl_filter.filter(ctrl)


class ReachingTask(BaseTask):
    """
    End-effectors track smooth reach trajectories via operational space control.
    No IK — Jacobian transpose maps task-space forces directly to joint torques.
    """

    def initialize(self, model, data):
        mujoco.mj_forward(model, data)
        self.ctrl_filter.reset()

        n_active = np.random.randint(1, max(2, len(self.ee_body_ids) + 1))
        self.active_ees = np.random.choice(
            self.ee_body_ids, size=n_active, replace=False
        ).tolist()

        ee_pos = self._get_ee_positions(data)
        self.trajectories = {}
        for bid in self.active_ees:
            self.trajectories[bid] = self.traj.reach_targets(
                ee_pos[bid],
                n_targets=np.random.randint(3, 7),
                radius=np.random.uniform(0.1, 0.35),
                move_time=np.random.uniform(0.8, 1.5),
                dt=self.dt,
            )

    def compute_control(self, model, data, step, total_steps):
        targets = {}
        for bid in self.active_ees:
            _, pos = self.trajectories[bid]
            targets[bid] = pos[min(step, len(pos) - 1)]

        tau = self.osc.compute(data, targets)
        ctrl = self.osc.to_ctrl(model, tau)
        return self._filtered_ctrl(ctrl)


class SteppingTask(BaseTask):
    """
    Locomotion-like motions: alternating foot targets with arm counter-swing.
    """

    def initialize(self, model, data):
        mujoco.mj_forward(model, data)
        self.ctrl_filter.reset()

        ee_pos = self._get_ee_positions(data)
        sorted_ees = sorted(ee_pos.items(), key=lambda x: x[1][2])
        n_feet = max(2, len(sorted_ees) // 2)
        self.foot_ees = [bid for bid, _ in sorted_ees[:n_feet]]
        self.arm_ees = [bid for bid in self.ee_body_ids if bid not in self.foot_ees]

        self.step_length = np.random.uniform(0.05, 0.25)
        self.step_height = np.random.uniform(0.03, 0.12)
        self.step_duration = np.random.uniform(0.5, 1.2)
        direction = np.random.uniform(-np.pi, np.pi)
        self.forward = np.array([np.cos(direction), np.sin(direction), 0.0])

        self.foot_trajs = {}
        for i, bid in enumerate(self.foot_ees):
            self.foot_trajs[bid] = self._step_seq(ee_pos[bid], 6, i)

        self.arm_trajs = {}
        for i, bid in enumerate(self.arm_ees):
            self.arm_trajs[bid] = self.traj.sinusoidal(
                ee_pos[bid],
                amplitude=np.array([0.1, 0.08, 0.03]),
                frequency=1.0 / (2 * self.step_duration),
                phase=np.pi * (i % 2),
                duration=6 * self.step_duration,
                dt=self.dt,
            )

    def _step_seq(self, start, n_steps, foot_idx):
        parts = []
        cur = start.copy()
        gz = start[2]
        lat = np.array([-self.forward[1], self.forward[0], 0.0])
        for si in range(n_steps):
            nf = int(self.step_duration / self.dt)
            if (si % len(self.foot_ees)) == foot_idx:
                nxt = cur + self.forward * self.step_length
                nxt += lat * 0.03 * ((-1) ** foot_idx)
                nxt[2] = gz
                # Use minimum-jerk for XY and parabolic for Z
                t_arr = np.linspace(0, 1, nf)
                s = 10*t_arr**3 - 15*t_arr**4 + 6*t_arr**5
                traj = np.zeros((nf, 3))
                traj[:, 0] = cur[0] + s * (nxt[0] - cur[0])
                traj[:, 1] = cur[1] + s * (nxt[1] - cur[1])
                traj[:, 2] = gz + self.step_height * 4 * s * (1 - s)
                parts.append(traj)
                cur = nxt.copy()
            else:
                parts.append(np.tile(cur, (nf, 1)))
        return np.concatenate(parts)

    def compute_control(self, model, data, step, total_steps):
        targets = {}
        for bid in self.foot_ees:
            traj = self.foot_trajs[bid]
            targets[bid] = traj[min(step, len(traj) - 1)]
        for bid in self.arm_ees:
            _, p = self.arm_trajs[bid]
            targets[bid] = p[min(step, len(p) - 1)]

        tau = self.osc.compute(data, targets)
        ctrl = self.osc.to_ctrl(model, tau)
        return self._filtered_ctrl(ctrl)


class COMTrackingTask(BaseTask):
    """
    Whole-body motions via COM Jacobian tracking.
    """

    def initialize(self, model, data):
        mujoco.mj_forward(model, data)
        self.ctrl_filter.reset()

        com = self.skeleton.get_com_position(data)
        ttype = np.random.choice(["squat"])
        dur = 6.0

        if ttype == "sway":
            a = np.random.uniform(0.03, 0.12)
            f = np.random.uniform(0.2, 0.6)
            _, self.com_traj = self.traj.sinusoidal(
                com, np.array([a, a, 0.01]),
                np.array([f, f*0.7, f*0.5]),
                np.array([0, np.pi/2, 0]), dur, self.dt,
            )
        elif ttype == "squat":
            az = np.random.uniform(0.03, 0.15)
            f = np.random.uniform(0.15, 0.5)
            _, self.com_traj = self.traj.sinusoidal(
                com, np.array([0.01, 0.01, az]),
                np.array([f*0.3, f*0.3, f]), duration=dur, dt=self.dt,
            )
        elif ttype == "circle":
            r = np.random.uniform(0.03, 0.1)
            f = np.random.uniform(0.2, 0.5)
            _, self.com_traj = self.traj.circular(
                com, r, np.random.choice(["xy", "xz"]), f, dur, self.dt,
            )
        else:
            _, self.com_traj = self.traj.random_waypoint_spline(
                com, 3, dur, 0.08, self.dt,
            )

        dists = [np.linalg.norm(data.xpos[b] - com) for b in range(1, model.nbody)]
        self.com_body = np.argmin(dists) + 1
        self.jacp = np.zeros((3, model.nv))

    def compute_control(self, model, data, step, total_steps):
        idx = min(step, len(self.com_traj) - 1)
        target_com = self.com_traj[idx]
        current_com = self.skeleton.get_com_position(data)
        com_err = target_com - current_com

        mujoco.mj_jacSubtreeCom(model, data, self.jacp, self.com_body)

        # COM velocity
        com_vel = self.jacp @ data.qvel

        # Task-space PD
        kp, kd = 60.0, 15.0
        f_task = kp * com_err - kd * com_vel

        # Clamp
        f_mag = np.linalg.norm(f_task)
        if f_mag > 30.0:
            f_task *= 30.0 / f_mag

        tau = self.jacp.T @ f_task + data.qfrc_bias
        ctrl = self.osc.to_ctrl(model, tau)
        return self._filtered_ctrl(ctrl)


class CoordinatedTask(BaseTask):
    """
    Multi-effector correlated motions via operational space control.
    """
    PATTERNS = ["anti_phase", "symmetric", "sequential", "wave", "coupled"]

    def initialize(self, model, data):
        mujoco.mj_forward(model, data)
        self.ctrl_filter.reset()

        ee_pos = self._get_ee_positions(data)
        centers = [ee_pos[bid] for bid in self.ee_body_ids]
        n = len(self.ee_body_ids)
        dur = 6.0

        pat = np.random.choice(self.PATTERNS)
        bf = np.random.uniform(0.2, 0.6)
        amp = np.random.uniform(0.08, 0.25)

        if pat == "anti_phase":
            _, self.ee_trajs = self.traj.correlated_multi(
                centers, bf, [1.0]*n,
                [np.pi*(i%2) for i in range(n)], amp, dur, self.dt,
            )
        elif pat == "symmetric":
            _, self.ee_trajs = self.traj.correlated_multi(
                centers, bf, [1.0]*n, [0.0]*n, amp, dur, self.dt,
            )
        elif pat == "sequential":
            _, self.ee_trajs = self.traj.correlated_multi(
                centers, bf, [1.0]*n,
                [2*np.pi*i/n for i in range(n)], amp, dur, self.dt,
            )
        elif pat == "wave":
            _, self.ee_trajs = self.traj.correlated_multi(
                centers, bf, [1.0+0.3*i for i in range(n)],
                [0.0]*n, amp, dur, self.dt,
            )
        else:
            self.ee_trajs = []
            for i in range(n):
                if i % 2 == 0:
                    _, p = self.traj.random_waypoint_spline(
                        centers[i], 3, dur, np.random.uniform(0.1, 0.25), self.dt,
                    )
                else:
                    _, p = self.traj.sinusoidal(
                        centers[i], np.random.uniform(0.08, 0.2),
                        np.random.uniform(0.2, 0.5),
                        np.random.uniform(0, 2*np.pi), dur, self.dt,
                    )
                self.ee_trajs.append(p)

    def compute_control(self, model, data, step, total_steps):
        targets = {}
        for i, bid in enumerate(self.ee_body_ids):
            t = self.ee_trajs[i]
            targets[bid] = t[min(step, len(t) - 1)]

        tau = self.osc.compute(data, targets)
        ctrl = self.osc.to_ctrl(model, tau)
        return self._filtered_ctrl(ctrl)


class RandomPrimitiveTask(BaseTask):
    """
    Joint-space exploration with chain-aware sinusoidal correlations.
    Tracks pre-computed smooth spline trajectories with PD control.
    """

    def initialize(self, model, data):
        mujoco.mj_forward(model, data)
        self.ctrl_filter.reset()

        q_init = data.qpos.copy()
        lower, upper = self.skeleton.get_all_joint_ranges()
        dur = 6.0
        nf = int(dur / self.dt)

        mode = np.random.choice(["sinusoidal_chains", "spline"])

        if mode == "sinusoidal_chains":
            self.q_traj = np.tile(q_init, (nf, 1))
            for chain in self.skeleton.kinematic_chains:
                joints = self.skeleton.get_joints_in_chain(chain)
                if not joints:
                    continue
                bf = np.random.uniform(0.15, 0.8)
                bp = np.random.uniform(0, 2 * np.pi)
                amp_s = np.random.uniform(0.2, 0.6)
                for ji, jinfo in enumerate(joints):
                    if jinfo.type == mujoco.mjtJoint.mjJNT_FREE:
                        continue
                    adr = jinfo.qpos_adr
                    po = ji * np.random.uniform(0.2, 0.6)
                    if jinfo.type in (mujoco.mjtJoint.mjJNT_HINGE,
                                       mujoco.mjtJoint.mjJNT_SLIDE):
                        lo, hi = lower[adr], upper[adr]
                        c = (lo + hi) / 2
                        hr = (hi - lo) / 2
                        t = np.arange(nf) * self.dt
                        f = bf * np.random.uniform(0.8, 1.2)
                        self.q_traj[:, adr] = c + hr * amp_s * np.sin(
                            2 * np.pi * f * t + bp + po
                        )
                    elif jinfo.type == mujoco.mjtJoint.mjJNT_BALL:
                        axis = np.random.randn(3)
                        axis /= np.linalg.norm(axis) + 1e-8
                        ma = np.random.uniform(0.2, 0.8)
                        t = np.arange(nf) * self.dt
                        f = bf * np.random.uniform(0.8, 1.2)
                        angles = ma * np.sin(2*np.pi*f*t + bp + po)
                        for fr in range(nf):
                            h = angles[fr] / 2
                            s = np.sin(h)
                            self.q_traj[fr, adr:adr+4] = [
                                np.cos(h), axis[0]*s, axis[1]*s, axis[2]*s
                            ]
            self.q_traj = np.clip(self.q_traj, lower, upper)
        else:
            _, self.q_traj = self.traj.joint_space_spline(
                q_init, (lower, upper),
                np.random.randint(3, 6), dur, self.dt,
                np.random.uniform(0.2, 0.5),
            )

    def compute_control(self, model, data, step, total_steps):
        idx = min(step, len(self.q_traj) - 1)
        ctrl = self.pd.compute(data, self.q_traj[idx])
        return self._filtered_ctrl(ctrl)


# =============================================================================
# Quaternion Helpers
# =============================================================================

def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


# =============================================================================
# Generator
# =============================================================================

class SyntheticMotionGenerator:
    """
    Generates synthetic motion data from a MuJoCo model.

    Args:
        model_path: Path to MuJoCo XML model file
        kp_scale: PD gain scale for joint-space tasks
        osc_kp: Operational space controller proportional gain
        osc_kd: Operational space controller derivative gain
        ee_body_names: Manual end-effector body names (auto-detected if None)
        seed: Random seed
    """

    def __init__(self, model_path: str, kp_scale: float = 1.0,
                 osc_kp: float = 50.0, osc_kd: float = 2.0,
                 ee_body_names: list[str] | None = None, seed: int = 42):
        np.random.seed(seed)
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.skeleton = SkeletonAnalyzer(self.model)
        self.pd = PDController(self.model, kp_scale=kp_scale)
        self.osc = OperationalSpaceController(self.model, kp=osc_kp, kd=osc_kd)
        self.traj = TrajectoryGenerator()

        if ee_body_names:
            self.ee_ids = [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
                for n in ee_body_names
            ]
        else:
            self.ee_ids = self.skeleton.detect_end_effectors()

    def generate(self, num_episodes: int = 50, episode_length: float = 4.0,
                 tasks: list[str] | None = None) -> list[dict]:
        """
        Generate synthetic motions across all task types.

        Args:
            num_episodes: Episodes per task
            episode_length: Duration per episode in seconds
            tasks: Task names to use (None = all)

        Returns:
            List of dicts with "qpos", "qvel", "xpos", "xquat", "task" keys.
        """
        task_map = {
            # "reaching": ReachingTask,
            # "stepping": SteppingTask,
            "com_tracking": COMTrackingTask,
            # "coordinated": CoordinatedTask,
            # "random_primitives": RandomPrimitiveTask,
        }

        if tasks is None:
            active_tasks = list(task_map.items())
        else:
            active_tasks = [(n, task_map[n]) for n in tasks if n in task_map]

        dt = self.model.opt.timestep
        steps = int(episode_length / dt)
        all_motions = []

        for task_name, task_cls in active_tasks:
            task = task_cls(
                self.model, self.skeleton, self.pd,
                self.osc, self.traj, self.ee_ids,
            )

            for ep in range(num_episodes):
                mujoco.mj_resetData(self.model, self.data)

                # Settle under gravity
                self.data.ctrl[:] = 0.0
                for _ in range(200):
                    mujoco.mj_step(self.model, self.data)
                self.data.qvel[:] = 0.0
                mujoco.mj_forward(self.model, self.data)

                task.initialize(self.model, self.data)

                qpos_buf, qvel_buf = [], []
                xpos_buf, xquat_buf = [], []
                unstable = 0

                for step in range(steps):
                    ctrl = task.compute_control(self.model, self.data, step, steps)
                    self.data.ctrl[:] = ctrl

                    mujoco.mj_step(self.model, self.data)

                    # Stability check
                    is_unstable = (
                        np.any(np.isnan(self.data.qpos)) or
                        np.any(np.isinf(self.data.qpos)) or
                        np.max(np.abs(self.data.qvel)) > 100.0
                    )
                    if is_unstable:
                        unstable += 1
                        if unstable > 3:
                            break
                        self.data.qvel[:] *= 0.1
                        self.data.ctrl[:] *= 0.1
                        mujoco.mj_forward(self.model, self.data)
                        # Hold last valid frame to preserve temporal continuity
                        if qpos_buf:
                            qpos_buf.append(qpos_buf[-1].copy())
                            qvel_buf.append(np.zeros_like(qvel_buf[-1]))
                            xpos_buf.append(xpos_buf[-1].copy())
                            xquat_buf.append(xquat_buf[-1].copy())
                        continue
                    unstable = 0

                    qpos_buf.append(self.data.qpos.copy())
                    qvel_buf.append(self.data.qvel.copy())
                    xpos_buf.append(self.data.xpos.copy())
                    xquat_buf.append(self.data.xquat.copy())

                if qpos_buf:
                    all_motions.append({
                        "qpos": np.array(qpos_buf),
                        "qvel": np.array(qvel_buf),
                        "xpos": np.array(xpos_buf),
                        "xquat": np.array(xquat_buf),
                        "task": task_name,
                    })

        return all_motions