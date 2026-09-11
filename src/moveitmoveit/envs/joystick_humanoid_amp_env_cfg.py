from __future__ import annotations

import os

from isaaclab.utils.configclass import configclass

from moveitmoveit.utils.paths import MOTIONS_DIR

from .humanoid_amp_env_cfg import HumanoidAmpEnvCfg


@configclass
class JoystickHumanoidAmpEnvCfg(HumanoidAmpEnvCfg):
    """Humanoid AMP environment config with a command-conditioned joystick task (base class)."""

    reward_type: str = "joystick"

    observation_space = 84

    # velocity command sampling ranges
    command_lin_vel_range: tuple[float, float] = (-1.5, 1.5)
    """Range (m/s) to sample the forward (base-frame x) linear velocity command from."""

    command_lat_vel_range: tuple[float, float] = (-1.0, 1.0)
    """Range (m/s) to sample the lateral (base-frame y) linear velocity command from."""

    command_ang_vel_range: tuple[float, float] = (-1.0, 1.0)
    """Range (rad/s) to sample the yaw rate command from."""

    command_lin_vel_scale: float = 3.0
    """Scale applied to the squared linear velocity-command tracking error: ``exp(-command_lin_vel_scale * error)``."""

    command_ang_vel_scale: float = 3.0
    """Scale applied to the squared angular velocity-command tracking error: ``exp(-command_ang_vel_scale * error)``."""

    command_resampling_strategy: str = "interval"
    """When to resample each env's velocity command.

    * reset: resample only when the env resets (a new command each episode).
    * interval: also resample every ``command_resampling_interval_steps`` steps during the episode.
    """

    command_resampling_interval_steps: int = 500
    """Number of steps between command resamples when ``command_resampling_strategy == "interval"``."""


@configclass
class JoystickHumanoidAmpWalkEnvCfg(JoystickHumanoidAmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_walk.npz")
