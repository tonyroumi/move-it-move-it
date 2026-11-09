"""Key-triggered behaviors: which one each env is running, and how far through it is.

Training and play run the same code here, so a policy is trained on exactly the latching it is
played with. Pure torch, so it is tested without a simulator.
"""

import torch

from .manifest import CommandIndex

LOCOMOTION = -1
"""The mode of an env that is running no behavior."""


class BehaviorManager:
    """Each env's mode: locomotion, or the behavior a key press started.

    A key starts its behavior only when pressed (its level going from released to held) while the env
    is in locomotion. The behavior then runs until `update` is told it is complete. Releasing, holding
    or pressing keys meanwhile changes nothing, and a key still held when it completes does not start
    it again. When keys are pressed in the same step, the lowest-numbered behavior wins.
    """

    def __init__(self, num_envs: int, num_behaviors: int, device: torch.device | str):
        self.num_behaviors = num_behaviors
        self.mode = torch.full((num_envs,), LOCOMOTION, dtype=torch.long, device=device)
        """LOCOMOTION, or the index of the running behavior. Shape is (N,)."""
        self.start_time = torch.zeros(num_envs, device=device)
        """t_start: the env time the running behavior started at, in seconds. Shape is (N,)."""
        self._held = torch.zeros(num_envs, num_behaviors, dtype=torch.bool, device=device)

    @property
    def active(self) -> torch.Tensor:
        """Envs running a behavior. Shape is (N,)."""
        return self.mode != LOCOMOTION

    def update(self, keys: torch.Tensor, complete: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Set each env's mode for this step: back to locomotion if its behavior is complete, then
        to a behavior whose key was just pressed, if the env is in locomotion.

        Args:
            keys: Each behavior's key level; above 0.5 is held. Shape is (N, num_behaviors).
            complete: Whether each env's running behavior has completed. Shape is (N,).
            time: Each env's current time, in seconds. Shape is (N,).

        Returns:
            Envs whose behavior started this step. Shape is (N,).
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

    def phase(self, time: torch.Tensor, duration: torch.Tensor) -> torch.Tensor:
        """phi(t) = min(1, (t - t_start) / duration) for envs running a behavior, 0 in locomotion.

        Args:
            time: Each env's current time, in seconds. Shape is (N,).
            duration: Each env's running behavior's duration, in seconds. Shape is (N,).
        """
        progress = ((time - self.start_time) / duration.clamp_min(1e-6)).clamp(0.0, 1.0)
        return torch.where(self.active, progress, torch.zeros_like(progress))

    def reset(self, env_ids: torch.Tensor, keys: torch.Tensor, mode: torch.Tensor, start_time: torch.Tensor) -> None:
        """Put envs in `mode`, a behavior started at `start_time` or LOCOMOTION. The keys' current
        levels count as already seen, so a key held across a reset does not start a behavior.

        Args:
            env_ids: Shape is (M,).
            keys: Those envs' key levels. Shape is (M, num_behaviors).
            mode: Shape is (M,).
            start_time: Shape is (M,).
        """
        self.mode[env_ids] = mode
        self.start_time[env_ids] = start_time
        self._held[env_ids] = keys > 0.5


def settled(
    phase: torch.Tensor, dof_velocities: torch.Tensor, min_phase: torch.Tensor, max_dof_speed: torch.Tensor
) -> torch.Tensor:
    """Stable-posture completion: at least `min_phase` through the behavior, with an RMS joint speed
    below `max_dof_speed`. The robot is just as still in its guard before a move, so `min_phase` is
    what keeps that from counting: the manifest sets it past the end of each clip's move.

    Args:
        phase: Shape is (N,).
        dof_velocities: Shape is (N, num_dofs).
        min_phase: Per env; infinite for behaviors that don't settle. Shape is (N,).
        max_dof_speed: Per env, in rad/s. Shape is (N,).
    """
    rms_speed = dof_velocities.square().mean(dim=-1).sqrt()
    return (phase >= min_phase) & (rms_speed < max_dof_speed)


def behavior_commands(
    commands: torch.Tensor, behavior_keys: list[int], mode: torch.Tensor, phase: torch.Tensor
) -> torch.Tensor:
    """The commands with the latched behavior in place of the raw keys, as the policy observes them.

    Each behavior's key slot reads 1 while that behavior runs, so together they are z(t), a one-hot
    that is all 0 in locomotion, whatever the key is doing. BEHAVIOR_PHASE reads phi(t). Key slots
    that trigger no behavior pass through.

    Args:
        commands: Shape is (N, COMMAND_DIM).
        behavior_keys: The key slot of each behavior, in behavior order.
        mode: Shape is (N,).
        phase: Shape is (N,).
    """
    observed = commands.clone()
    if behavior_keys:
        one_hot = torch.nn.functional.one_hot(mode.clamp_min(0), len(behavior_keys)).to(commands.dtype)
        observed[:, behavior_keys] = one_hot * (mode != LOCOMOTION).unsqueeze(-1)
    observed[:, CommandIndex.BEHAVIOR_PHASE] = phase
    return observed


class RandomKeyPresses:
    """Training's stand-in for a person at the keyboard. It writes raw key levels, which the env
    latches with the same BehaviorManager it uses in play.

    An env whose keys have been released for at least a step presses one at `rate` presses per
    second, picking behavior b with probability `weights[b]`, and holds it for 1 to
    `max_hold_steps` steps. Taps, holds past a behavior's end and presses while one runs all occur.
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
