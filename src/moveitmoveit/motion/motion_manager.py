import torch

from moveitmoveit.mdp.commands import COMMAND_DIM, CommandIndex
from moveitmoveit.motion.behavior import RandomKeyPresses
from moveitmoveit.motion.manifest import Manifest
from moveitmoveit.motion.specs import MotionSpec
from moveitmoveit.state import BodyState

LOCOMOTION = -1


def clip_phase(clip_times: torch.Tensor, duration: torch.Tensor) -> torch.Tensor:
    return (clip_times / duration.clamp_min(1e-6)).clamp(0.0, 1.0)


class MotionManager:
    """The per-env motion state manager"""

    def __init__(
        self,
        manifest: Manifest,
        num_envs: int,
        device: torch.device | str,
        step_dt: float = 1.0,
    ):
        self.manifest = manifest

        self._key_presses = RandomKeyPresses(
            num_envs,
            self.manifest.trigger_rate,
            self.manifest.trigger_weights,
            step_dt,
            device
        )
        # the locomotion motions an env that resets into a behavior returns to, by manifest weight
        self._locomotion_weights = torch.where(
            self.manifest.motion_behavior == LOCOMOTION, self.manifest.sample_prob, torch.zeros_like(self.manifest.sample_prob)
        )

        self.commands = torch.zeros(num_envs, COMMAND_DIM, device=device)
        self.motion_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.base_motion_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.clip_offsets = torch.zeros(num_envs, device=device)
        self.mode = torch.full((num_envs,), LOCOMOTION, dtype=torch.long, device=device)
        self.start_time = torch.zeros(num_envs, device=device)
        self.motion_phase = torch.zeros(num_envs, device=device)
        self.behaviors = torch.zeros(num_envs, self.manifest.num_behaviors, device=device)

        self._held = torch.zeros(num_envs, self.manifest.num_behaviors, dtype=torch.bool, device=device)

    @property
    def active(self) -> torch.Tensor:
        """Envs running a behavior."""
        return self.mode != LOCOMOTION

    def reset(self, env_ids: torch.Tensor, motion_ids: torch.Tensor, times: torch.Tensor):
        """Start env_ids in motion_ids at their clip times and release their behavior keys. """
        mode = self.manifest.motion_behavior[motion_ids]
        is_behavior = mode != LOCOMOTION

        base_motion_ids = motion_ids.clone()
        if not self.manifest.always_on and is_behavior.any():
            num_behaviors = int(is_behavior.sum())
            base_motion_ids[is_behavior] = torch.multinomial(self._locomotion_weights, num_behaviors, replacement=True)

        self.mode[env_ids] = mode
        self.motion_ids[env_ids] = motion_ids
        self.base_motion_ids[env_ids] = base_motion_ids
        # Set the start time and clip offset for behaviors and locomotion respectively.
        zeros = torch.zeros_like(times)
        self.start_time[env_ids] = torch.where(is_behavior, -times, zeros)
        self.clip_offsets[env_ids] = torch.where(is_behavior, zeros, times)

        # Restart all key presses
        self._held[env_ids] = False
        self._key_presses.reset(env_ids)

    def resample_commands(self, env_ids: torch.Tensor, state: BodyState) -> None:
        """Sample base-motion commands and anchor goal offsets at the sampling position once."""
        commands = self.manifest.sample_commands(self.base_motion_ids[env_ids])
        commands[:, CommandIndex.GOAL_X : CommandIndex.GOAL_Y + 1] += state.root_positions[env_ids, :2]
        self.commands[env_ids] = commands

    def set_commands(self, commands: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Store externally supplied commands; goal coordinates are already world-space targets."""
        if env_ids is None:
            self.commands[:] = commands
        else:
            self.commands[env_ids] = commands

    def step(self, time: torch.Tensor, resample_cmds: bool) -> None:
        """Advance each env's motion to episode `time`: complete running behaviors at phase 1, start
        behaviors, and record the phase and behavior the policy observes. """
        complete = (self.phase(time) >= 1.0)

        if self.manifest.always_on:
            # no locomotion to return to, so every env is running a behavior at every step
            restart = complete.nonzero(as_tuple=False).squeeze(-1)
            if restart.numel() > 0:
                self.start(restart, self.manifest.sample_behaviors(restart.numel()), time[restart])
        else:
            keys = self._key_presses.step() if resample_cmds else self.commands[:, self.manifest.behavior_keys]
            self.update(keys, complete, time)

        self.motion_ids = self._running_motion_ids()
        self.motion_phase = self.phase(time)
        behavior_indexes = torch.arange(self.manifest.num_behaviors, device=self.mode.device)
        self.behaviors = (self.mode.unsqueeze(-1) == behavior_indexes).float()

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

    def start(self, env_ids: torch.Tensor, behaviors: torch.Tensor, start_time: torch.Tensor) -> None:
        """Run `behaviors` in `env_ids` with no key press, on clip clocks that read 0 at `start_time`."""
        self.mode[env_ids] = behaviors
        self.start_time[env_ids] = start_time

    def _running_motion_ids(self) -> torch.Tensor:
        """The motion each env runs: its behavior's motion while one is running, else its base motion."""
        behavior_motion_ids = self.manifest.behavior_motion_ids
        if behavior_motion_ids.numel() == 0:
            return self.base_motion_ids.clone()
        behavior_ids = behavior_motion_ids[self.mode.clamp_min(0)]
        return torch.where(self.active, behavior_ids, self.base_motion_ids)

    def active_specs(self) -> list[MotionSpec]:
        """The spec of the motion each env runs, whose rewards and terminations apply."""
        return self.manifest.specs_for(self.motion_ids)

    def current_phase(self) -> torch.Tensor:
        """Each env's behavior phase, as the last `step` recorded it."""
        return self.motion_phase

    def current_behaviors(self) -> torch.Tensor:
        """Each env's running behavior as a one-hot encoding"""
        return self.behaviors

    def current_goals(self) -> torch.Tensor:
        """Goal activation for the currently running motions, not their locomotion bases."""
        return self.manifest.has_goal[self.motion_ids]

    def phase(self, time: torch.Tensor) -> torch.Tensor:
        progress = clip_phase(self.clip_times(time), self.manifest.durations[self.motion_ids])
        return torch.where(self.active, progress, torch.zeros_like(progress))

    def clip_times(self, time: torch.Tensor) -> torch.Tensor:
        """Each env's time at episode `time` in the clip it tracks
        """
        duration = self.manifest.durations[self.motion_ids]
        locomotion_times = self.clip_offsets + time
        wrapped = torch.remainder(locomotion_times, duration.clamp_min(1e-6))
        locomotion_times = torch.where(self.manifest.loop[self.motion_ids], wrapped, locomotion_times)
        return torch.where(self.active, time - self.start_time, locomotion_times)
