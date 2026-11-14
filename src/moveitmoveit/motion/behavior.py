"""Key-triggered behaviors: which one each env is running, and how far through it is."""

import torch

from moveitmoveit.motion.manifest import CommandIndex

LOCOMOTION = -1


class BehaviorManager:
    """Each env's mode: locomotion, or the behavior a key press, or `start`, started."""

    def __init__(self, num_envs: int, num_behaviors: int, device: torch.device | str):
        self.num_behaviors = num_behaviors
        self.mode = torch.full((num_envs,), LOCOMOTION, dtype=torch.long, device=device)
        self.start_time = torch.zeros(num_envs, device=device)
        self._held = torch.zeros(num_envs, num_behaviors, dtype=torch.bool, device=device)

    @property
    def active(self) -> torch.Tensor:
        """Envs running a behavior."""
        return self.mode != LOCOMOTION

    def update(self, keys: torch.Tensor, complete: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Set each env's mode for this step: back to locomotion if its behavior is complete, then
        to a behavior whose key was just pressed, if the env is in locomotion.
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

    def start(self, envs: torch.Tensor, behaviors: torch.Tensor, start_time: torch.Tensor) -> None:
        """Run `behaviors` in `envs` (indices or a mask) with no key press, on clip clocks that read
        0 at `start_time`. A start time before the episode's puts the env part-way through the clip.
        """
        self.mode[envs] = behaviors
        self.start_time[envs] = start_time

    def phase(self, time: torch.Tensor, duration: torch.Tensor) -> torch.Tensor:
        """phi(t) = min(1, (t - t_start) / duration) for envs running a behavior, 0 in locomotion.
        """
        progress = ((time - self.start_time) / duration.clamp_min(1e-6)).clamp(0.0, 1.0)
        return torch.where(self.active, progress, torch.zeros_like(progress))

    def reset(self, env_ids: torch.Tensor, held_keys: torch.Tensor) -> None:
        """Put envs in locomotion with no behavior running. `held_keys` count as already seen, so a
        key still held across the reset does not start a behavior.
        """
        self.mode[env_ids] = LOCOMOTION
        self.start_time[env_ids] = 0.0
        self._held[env_ids] = held_keys > 0.5


def settled(
    phase: torch.Tensor, dof_velocities: torch.Tensor, min_phase: torch.Tensor, max_dof_speed: torch.Tensor
) -> torch.Tensor:
    """Stable-posture completion"""
    rms_speed = dof_velocities.square().mean(dim=-1).sqrt()
    return (phase >= min_phase) & (rms_speed < max_dof_speed)


def reference_times(
    time: torch.Tensor,
    behaviors: BehaviorManager,
    locomotion_start_times: torch.Tensor,
    duration: torch.Tensor,
    loop: torch.Tensor,
) -> torch.Tensor:
    """Each env's time in the clip it tracks, which `MotionLibrary` clamps to [0, duration].

    A running behavior's clip started with it, at t_start, and is never wrapped: the behavior
    completes on this same unwrapped clock, at phase 1 or when it settles, and returns to
    locomotion. A locomotion clip's clock started at `locomotion_start_times` and wraps modulo its
    duration where `loop`, so its target repeats; elsewhere it runs on and holds the last frame.
    """
    locomotion_times = locomotion_start_times + time
    wrapped = torch.remainder(locomotion_times, duration.clamp_min(1e-6))
    locomotion_times = torch.where(loop, wrapped, locomotion_times)
    return torch.where(behaviors.active, time - behaviors.start_time, locomotion_times)


def behavior_commands(
    commands: torch.Tensor, behavior_keys: list[int], mode: torch.Tensor, phase: torch.Tensor
) -> torch.Tensor:
    """The commands with the behavior id."""
    observed = commands.clone()
    if behavior_keys:
        one_hot = torch.nn.functional.one_hot(mode.clamp_min(0), len(behavior_keys)).to(commands.dtype)
        observed[:, behavior_keys] = one_hot * (mode != LOCOMOTION).unsqueeze(-1)
    observed[:, CommandIndex.BEHAVIOR_PHASE] = phase
    return observed


class RandomKeyPresses:
    """Training's stand-in for a person at the keyboard. It writes raw key levels, which the env
    latches with the same BehaviorManager it uses in play.
    """

    def __init__(
        self,
        num_envs: int,
        rate: float,
        weights: torch.Tensor,
        max_hold_steps: int,
        step_dt: float,
        device: torch.device | str,
    ):
        self._press_prob = rate * step_dt
        self._weights = weights
        self._max_hold_steps = max_hold_steps
        self._behavior = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._hold = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._was_held = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def step(self) -> torch.Tensor:
        """This step's key levels. Shape is (N, num_behaviors)."""
        num_envs, device = self._hold.shape[0], self._hold.device
        levels = torch.zeros(num_envs, self._weights.shape[0], device=device)
        if self._press_prob <= 0.0:
            return levels

        press = (self._hold == 0) & ~self._was_held & (torch.rand(num_envs, device=device) < self._press_prob)
        if press.any():
            num_presses = int(press.sum())
            self._behavior[press] = torch.multinomial(self._weights, num_presses, replacement=True)
            self._hold[press] = torch.randint(1, self._max_hold_steps + 1, (num_presses,), device=device)

        held = self._hold > 0
        levels[held, self._behavior[held]] = 1.0
        self._hold = (self._hold - 1).clamp_min(0)
        self._was_held = held
        return levels

    def reset(self, env_ids: torch.Tensor) -> None:
        """Release these envs' keys."""
        self._hold[env_ids] = 0
        self._was_held[env_ids] = False
