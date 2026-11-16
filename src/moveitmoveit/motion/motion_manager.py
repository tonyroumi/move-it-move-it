import torch

from moveitmoveit.mdp import BodyState
from moveitmoveit.motion.behavior import RandomKeyPresses, clip_phase, settled
from moveitmoveit.motion.library import MotionFrames, MotionLibrary
from moveitmoveit.motion.motion_set import MotionSet
from moveitmoveit.motion.specs import MotionSpec
from moveitmoveit.utils.transforms import apply_random_yaw

LOCOMOTION = -1


class MotionManager:
    """The per-env motion state manager"""

    def __init__(
        self,
        manifest: MotionSet,
        motion_durations: torch.Tensor,
        num_envs: int,
        device: torch.device | str,
        step_dt: float = 1.0,
    ):
        self.manifest = manifest

        self._key_presses = RandomKeyPresses(
            num_envs,
            self.manifest.trigger_rate,
            self.manifest.trigger_weights,
            max(1, round(self.cfg.trigger_max_hold_s / self.step_dt)),
            step_dt,
            device
        )

        self.motion_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.base_motion_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.clip_offsets = torch.zeros(num_envs, device=device)
        self.mode = torch.full((num_envs,), LOCOMOTION, dtype=torch.long, device=device)
        self.start_time = torch.zeros(num_envs, device=device)
        self.motion_phase = torch.zeros(num_envs, device=device)
        self.behaviors = torch.zeros(num_envs, motions.num_behaviors, device=device)
        self.motion_durations = motion_durations

        self._held = torch.zeros(num_envs, motions.num_behaviors, dtype=torch.bool, device=device)

    @property
    def active(self) -> torch.Tensor:
        """Envs running a behavior."""
        return self.mode != LOCOMOTION
    
    def observations(self) -> torch.Tensor:
        return torch.cat(
            [
                one_hot(self.behaviors),
                self.phase()
            ]
        )

    def reset(self, env_ids: torch.Tensor, behavior_ids: torch.Tensor, motion_ids: torch.Tensor, times: torch.Tensor):
        """Reset motions and release behavior keys for `env_ids`, returning their clip state without yaw.

        Every motion is eligible, using its manifest weight. A sampled behavior starts active at
        its sampled clip offset. A held key can trigger again after reset.
        """
        self.motion_ids[env_ids] = motion_ids
        self.base_motion_ids[env_ids] = motion_ids
        self.motion_start_times[env_ids] = times

        self.behaviors[env_ids] = behavior_ids

        # Restart all key presses
        self._key_presses.reset(env_ids)

    def step(self, time: torch.Tensor) -> None:
        complete = (self.phase(time) >= 1.0) 

        if self.motions.always_on:
            # no locomotion to return to, so every env is running a behavior at every step
            restart = complete.nonzero(as_tuple=False).squeeze(-1)
            if restart.numel() > 0:
                self.start(restart, self.sample_behaviors(restart.numel()), time[restart])
        else:
            self.update(self._key_presses.step(), complete, time)

        self.motion_ids = self._running_motion_ids()

    def update(self, keys: torch.Tensor, complete: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Set each env's mode for this step: back to locomotion if its behavior is complete, then
        to a behavior whose key was just pressed, if the env is in locomotion. Returns the envs a
        press started.
        """
        held = keys > 0.5
        pressed = held & ~self._held
        self._held = held

        self.mode[self.active & complete] = LOCOMOTION

        start = ~self.active & pressed.any(dim=-1)
        if start.any():
            # argmax returns the first maximum, i.e. the lowest pressed behavior
            self.mode[start] = pressed[start].int().argmax(dim=-1)
            self.start_time[start] = time[start]
        return start

    def _running_motion_ids(self) -> torch.Tensor:
        """The motion each env runs: its behavior's motion while one is running, else its base motion."""
        behavior_motion_ids = self.motions.behavior_motion_ids
        if behavior_motion_ids.numel() == 0:
            return self.base_motion_ids.clone()
        behavior_ids = behavior_motion_ids[self.mode.clamp_min(0)]
        return torch.where(self.active, behavior_ids, self.base_motion_ids)


    def phase(self, time: torch.Tensor) -> torch.Tensor:
        """phi(t) = min(1, (t - t_start) / duration) for envs running a behavior, 0 in locomotion:
        `clip_phase` on the behavior's clip clock, which decides when it completes.
        """
        progress = clip_phase(time - self.start_time, self.duration[self.motion_ids])
        return torch.where(self.active, progress, torch.zeros_like(progress))

    def clip_times(self, time: torch.Tensor) -> torch.Tensor:
        """Each env's time at episode `time` in the clip it tracks, which `MotionLibrary` clamps to
        [0, duration].

        A running behavior's clip started with it, at t_start, and is never wrapped: the behavior
        completes on this same unwrapped clock, at phase 1 or when it settles, and returns to
        locomotion. A locomotion clip's clock started at `clip_offsets` and wraps modulo its
        duration where the motion loops, so its target repeats; elsewhere it runs on and holds the
        last frame.
        """
        duration = self.duration[self.motion_ids]
        locomotion_times = self.clip_offsets + time
        wrapped = torch.remainder(locomotion_times, duration.clamp_min(1e-6))
        locomotion_times = torch.where(self.motions.loop[self.motion_ids], wrapped, locomotion_times)
        return torch.where(self.active, time - self.start_time, locomotion_times)