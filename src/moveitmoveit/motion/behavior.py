"""Behavior helpers: stable-posture completion, clip phase, and training's random key presses.
`MotionManager` runs the behaviors themselves."""

import torch




class RandomKeyPresses:
    """Training's stand-in for a person at the keyboard. It writes raw key levels, which the env
    latches with the same MotionManager it uses in play.
    """

    def __init__(
        self,
        num_envs: int,
        rate: float,
        weights: torch.Tensor,
        step_dt: float,
        device: torch.device | str,
    ):
        self._press_prob = rate * step_dt
        self._weights = weights
        self._behavior = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._hold = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._was_held = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def step(self) -> torch.Tensor:
        """This step's key levels"""
        num_envs, device = self._hold.shape[0], self._hold.device
        levels = torch.zeros(num_envs, self._weights.shape[0], device=device)
        if self._press_prob <= 0.0:
            return levels

        press = (self._hold == 0) & ~self._was_held & (torch.rand(num_envs, device=device) < self._press_prob)
        if press.any():
            num_presses = int(press.sum())
            self._behavior[press] = torch.multinomial(self._weights, num_presses, replacement=True)
            self._hold[press] = torch.randint(1, 3, (num_presses,), device=device)

        held = self._hold > 0
        levels[held, self._behavior[held]] = 1.0
        self._hold = (self._hold - 1).clamp_min(0)
        self._was_held = held
        return levels

    def reset(self, env_ids: torch.Tensor) -> None:
        """Release these envs' keys."""
        self._hold[env_ids] = 0
        self._was_held[env_ids] = False
