import mujoco
import numpy as np
import imageio
from scipy.interpolate import CubicSpline


class MujocoMotionGenerator:
    def __init__(
        self,
        xml_path: str,
        fps: int = 60,
        clip_seconds: float = 10.0,
        keyframe_spacing: int = 30,
        joint_limit_scale: float = 0.5,
        kp: float = 50.0,
        kd: float = 5.0,
        max_qvel: float = 50.0,
        min_root_height: float = 0.4,
        width: int = 1280,
        height: int = 720,
        camera_name: str | None = None,
    ):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.fps = fps
        self.dt = 1.0 / fps
        self.T = int(clip_seconds * fps)

        self.keyframe_spacing = keyframe_spacing
        self.joint_limit_scale = joint_limit_scale

        self.kp = kp
        self.kd = kd

        self.max_qvel = max_qvel
        self.min_root_height = min_root_height

        self.width = width
        self.height = height
        self.camera_name = camera_name

        self.joint_ids = self._get_actuated_joints()
        self.joint_ranges = self._get_joint_ranges()

        self.q_target_traj = None

        self.renderer = mujoco.Renderer(self.model, width, height)

    # ---------------------------------------------------
    # Model introspection
    # ---------------------------------------------------
    def _get_actuated_joints(self):
        return [
            j for j in range(self.model.njnt)
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        ]

    def _get_joint_ranges(self):
        ranges = {}
        for j in self.joint_ids:
            lo, hi = self.model.jnt_range[j]
            mid = 0.5 * (lo + hi)
            span = (hi - lo) * self.joint_limit_scale
            ranges[j] = (mid - span / 2, mid + span / 2)
        return ranges

    # ---------------------------------------------------
    # Motion generation
    # ---------------------------------------------------

    def _motor_control(self, q_target):
        qpos = self.data.qpos
        qvel = self.data.qvel

        ctrl = np.zeros(self.model.nu)

        for a in range(self.model.nu):
            j = self.model.actuator_trnid[a, 0]

            # Skip root joint
            if self.model.jnt_bodyid[j] == 0:
                continue

            qadr = self.model.jnt_qposadr[j]
            vadr = self.model.jnt_dofadr[j]

            pos_err = q_target[qadr] - qpos[qadr]
            vel_err = -qvel[vadr]

            u = self.kp * pos_err + self.kd * vel_err

            gear = self.model.actuator_gear[a][0]
            u /= max(gear, 1e-6)

            lo, hi = self.model.actuator_ctrlrange[a]
            ctrl[a] = np.clip(u, lo, hi)

        return ctrl

    def _sample_keyframes(self):
        times, keyframes = [], []

        for t in range(0, self.T, self.keyframe_spacing):
            q = self.data.qpos.copy()  # nq-sized

            for j, (lo, hi) in self.joint_ranges.items():
                adr = self.model.jnt_qposadr[j]

                # hinge joints are size 1 in qpos
                q[adr] = np.random.uniform(lo, hi)

            times.append(t)
            keyframes.append(q)

        return np.array(times), np.array(keyframes)

    def _build_trajectory(self):
        times, keyframes = self._sample_keyframes()
        spline = CubicSpline(times, keyframes, axis=0)
        self.q_target_traj = spline(np.arange(self.T))

    # ---------------------------------------------------
    # Control
    # ---------------------------------------------------
    def _pd_control(self, q_target):
        qpos = self.data.qpos[: self.model.nv]
        qvel = self.data.qvel[: self.model.nv]
        return self.kp * (q_target[1:] - qpos) - self.kd * qvel

    # ---------------------------------------------------
    # Safety
    # ---------------------------------------------------
    def _unsafe(self):
        if np.any(np.isnan(self.data.qpos)):
            return True
        if np.any(np.abs(self.data.qvel) > self.max_qvel):
            return True
        if self.data.qpos[2] < self.min_root_height:
            return True
        return False

    def _reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._build_trajectory()

    # ---------------------------------------------------
    # Offscreen rendering to MP4
    # ---------------------------------------------------
    def render_to_mp4(self, output_path: str):
        self._reset()

        writer = imageio.get_writer(
            output_path,
            fps=self.fps,
            codec="libx264",
            quality=8,
        )

        try:
            for t in range(self.T):
                if self._unsafe():
                    break

                q_target = self.q_target_traj[t]
                tau = self._pd_control(q_target)

                ctrl = self._motor_control(q_target)
                self.data.ctrl[:] = ctrl
                mujoco.mj_step(self.model, self.data)

                if self.camera_name is None:
                    self.renderer.update_scene(self.data)
                else:
                    self.renderer.update_scene(self.data, camera=self.camera_name)
                frame = self.renderer.render()
                writer.append_data(frame)

        finally:
            writer.close()
