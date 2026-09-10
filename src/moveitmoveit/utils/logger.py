from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Tuple
import collections
import time

import numpy as np
import torch


class Logger:
    """ RL logger with optional TensorBoard and/or Weights & Biases backends.

    In addition to generic scalar/histogram/video logging, this class owns all
    training diagnostics: episode reward/length tracking, agent update
    diagnostics, and iteration timing (collection/learning time, ETA).

    Agents push data in via `add_env_info` + `env_step` (per env step) and
    `add_info` / `add_info` + `grad_step` (per gradient step); the
    runner brackets each phase with the `timing` context manager and calls
    `step` once per env step. `log` then flushes everything (all tracked tags
    go to tensorboard/wandb via `write_data`; only a curated "core" subset is
    printed to stdout) and checkpoints the agent when a new best mean episode
    reward is reached.
    """

    # Training diagnostics considered "core" enough to print to the screen.
    _CORE_TRAIN_KEYS = ("Total Loss", "Policy Loss", "Value Loss", "Discriminator Loss")

    def __init__(
        self,
        backend: str = "tensorboard",
        log_dir: str = "logs",
        write_interval: int = 1,
        checkpoint_interval: int = 1000,
        total_timesteps: Optional[int] = None,
        wandb_kwargs: Optional[Dict[str, Any]] = None,
    ):
        assert backend in ("tensorboard", "wandb", "both", "none"), (
            f"Unknown backend: {backend}"
        )
        self.backend = backend
        self.log_dir = log_dir
        self.total_timesteps = total_timesteps
        self.write_interval = write_interval
        self.checkpoint_interval = checkpoint_interval

        self._tb_writer = None
        self._wandb = None

        use_tb = backend in ("tensorboard", "both")
        use_wandb = backend in ("wandb", "both")

        if use_tb:
            from torch.utils.tensorboard import SummaryWriter

            self._tb_writer = SummaryWriter(log_dir=log_dir)

        if use_wandb:
            import wandb

            wandb.init(
                **(wandb_kwargs or {}),
            )
            self._wandb = wandb

        self.timestep = 0
        self._iteration = 0
        self._metric_step = collections.defaultdict(int)

        self._tracking_data = collections.defaultdict(float)
        self._tracking_step = collections.defaultdict(int)

        # Episode reward/length tracking.
        self._track_rewards = collections.deque(maxlen=100)
        self._track_timesteps = collections.deque(maxlen=100)
        self._cumulative_rewards = None
        self._cumulative_timesteps = None
        self.mean_episode_reward: Optional[float] = None
        self._best_mean_episode_reward = float("-inf")

        # Per-episode reward term tracking (mirrors the episode reward tracking above,
        # keyed by term name), staged from `info["reward_terms"]`.
        self._track_reward_terms: Dict[str, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=100)
        )
        self._cumulative_reward_terms: Dict[str, torch.Tensor] = {}

        # Per-step env info staged by `add_env_info`, consumed by `env_step`.
        self._env_info: Dict[str, Any] = {}

        # Per-update training diagnostics staged by `add_info`/`add_info`,
        # averaged and flushed by `log`.
        self._update_diagnostics: Dict[str, list] = collections.defaultdict(list)
        self._step_tracker : Dict[str, int] = collections.defaultdict(int)

        # Named wall-clock timings recorded by `timing`, e.g. "Collection Time".
        self._start_time = time.time()
        self._diagnostics: Dict[str, float] = {}

        # Curated values shown by `log`, refreshed by `env_step` / `log`.
        self._core_performance: Dict[str, float] = {}
        self._core_train: Dict[str, float] = {}
        self._core_rewards: Dict[str, float] = {}

    def step(self, num_steps: int = 1) -> None:
        """Advance the env-step counter used as the x-axis for step-indexed tags."""
        self.timestep += num_steps

    def step_metric(self, idx: int = 0) -> None:
        """Advance the metric step counter used as the x-axis for `Train/*` tags."""
        self._metric_step[idx] += 1

    @contextmanager
    def timing(self, name: str):
        """Time a block of code and record it under `name` (e.g. "Collection Time")."""
        start = time.perf_counter()

        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._diagnostics[name] = elapsed
            self.track_data(f"Performance/{name} [s]", elapsed, self.timestep)

    def track_data(self, tag: str, value: float, step: int):
        self._tracking_data[tag] = value
        self._tracking_step[tag] = step

    def add_env_info(self, infos: dict) -> None:
        """Stage the current step's reward/done/info payload for `env_step` to consume."""
        self._env_info = infos

    def env_step(self) -> None:
        """Accumulate per-env reward/length and log episode-completion statistics."""
        info = self._env_info
        rewards = info["rewards"]
        dones = info["dones"]
        step_dt = info.get("step_dt")
        timestep = self.timestep

        if self._cumulative_rewards is None:
            self._cumulative_rewards = torch.zeros_like(rewards, dtype=torch.float32)
            self._cumulative_timesteps = torch.zeros_like(rewards, dtype=torch.int32)

        self._cumulative_rewards.add_(rewards)
        self._cumulative_timesteps.add_(1)

        if dones.any():
            self._track_rewards.extend(self._cumulative_rewards[dones].tolist())
            self._track_timesteps.extend(self._cumulative_timesteps[dones].tolist())

            # reset the cumulative rewards and timesteps
            self._cumulative_rewards[dones] = 0
            self._cumulative_timesteps[dones] = 0

        if len(self._track_rewards):
            track_rewards = np.array(self._track_rewards)
            track_timesteps = np.array(self._track_timesteps)

            self.mean_episode_reward = float(np.mean(track_rewards))

            self.track_data("Performance/Episode Reward (max)", np.max(track_rewards), timestep)
            self.track_data("Performance/Episode Reward (min)", np.min(track_rewards), timestep)
            self.track_data("Performance/Episode Reward (mean)", self.mean_episode_reward, timestep)

            self.track_data("Performance/Episode Length (max)", np.max(track_timesteps), timestep)
            self.track_data("Performance/Episode Length (min)", np.min(track_timesteps), timestep)
            self.track_data("Performance/Episode Length (mean)", np.mean(track_timesteps), timestep)

            self._core_performance = {
                "Episode Reward (mean)": self.mean_episode_reward,
                "Episode Reward (max)": float(np.max(track_rewards)),
                "Episode Reward (min)": float(np.min(track_rewards)),
                "Episode Length (mean)": float(np.mean(track_timesteps)),
            }

            if step_dt is not None:
                mean_episode_time = float(np.mean(track_timesteps) * step_dt)
                self.track_data("Performance/Episode Time (mean) [s]", mean_episode_time, timestep)
                self._core_performance["Episode Time (mean) [s]"] = mean_episode_time

        for k, v in info.get("log", {}).items():
            self.track_data(tag=k, value=v, step=timestep)

        core_rewards = {}
        for name, values in info.get("reward_terms", {}).items():
            if name not in self._cumulative_reward_terms:
                self._cumulative_reward_terms[name] = torch.zeros_like(values, dtype=torch.float32)
            self._cumulative_reward_terms[name].add_(values)

            if dones.any():
                self._track_reward_terms[name].extend(self._cumulative_reward_terms[name][dones].tolist())
                self._cumulative_reward_terms[name][dones] = 0

            if len(self._track_reward_terms[name]):
                mean_value = float(np.mean(self._track_reward_terms[name]))
                self.track_data(f"Reward/{name} (mean)", mean_value, timestep)
                core_rewards[f"{name} (mean)"] = mean_value

        if core_rewards:
            self._core_rewards = core_rewards

    def add_info(self, name: str, value, grad_num: int = 0) -> None:
        """Record a diagnostic."""
        self._add_diagnostic(name, value, grad_num=grad_num)

    def _add_diagnostic(self, name: str, value, grad_num: int = 0) -> None:
        self._update_diagnostics[name].append(value)
        self._step_tracker[name] = grad_num

    def _flush_update_diagnostics(self) -> None:
        core = {}
        for name, values in self._update_diagnostics.items():
            value = sum(values) / len(values)
            assoc_grad_num = self._step_tracker[name]
            step = self._metric_step[assoc_grad_num]
            self.track_data(f"Train/{name}", value, step)
            self.track_data(f"Debug /{name}", value, self.timestep)

            if name in self._CORE_TRAIN_KEYS:
                core[name] = value

        self._core_train = core
        self._update_diagnostics = collections.defaultdict(list)

    def log(self, write_checkpoint: Optional[Callable[..., None]] = None) -> None:
        """Flush diagnostics, checkpoint if warranted, and print the core summary to stdout."""
        self._flush_update_diagnostics()
        self._iteration += 1

        if self._iteration % self.write_interval == 0:
            self.write_data()

        if self._iteration % self.checkpoint_interval == 0:
            if write_checkpoint is not None:
                write_checkpoint(self.timestep)

        if write_checkpoint is not None and self.mean_episode_reward is not None:
            if self.mean_episode_reward > self._best_mean_episode_reward:
                self._best_mean_episode_reward = self.mean_episode_reward
                write_checkpoint(self.timestep, filename="best_agent.pt")

        width = 60
        lines = [f"\n{f' Iteration {self._iteration} ':-^{width}}"]

        timestep_str = f"{self.timestep}/{self.total_timesteps}" if self.total_timesteps else f"{self.timestep}"
        lines.append(f"{'Timestep:':<24}{timestep_str}")
        lines.append(f"{'Collection time:':<24}{self._diagnostics.get('Collection Time', 0.0):.2f}s")
        lines.append(f"{'Learning time:':<24}{self._diagnostics.get('Learning Time', 0.0):.2f}s")
        if "Learning Rate" in self._core_train:
            lines.append(f"{'Learning rate:':<24}{self._core_train['Learning Rate']:.2e}")
        lines.append(f"{'ETA:':<24}{self._eta()}")

        if self._core_performance:
            lines.append("-" * width)
            for k, v in self._core_performance.items():
                lines.append(f"{k + ':':<24}{v:.4f}")

        if self._core_rewards:
            lines.append("-" * width)
            for k, v in self._core_rewards.items():
                lines.append(f"{k + ':':<24}{v:.4f}")

        core_train_display = {k: v for k, v in self._core_train.items() if k != "Learning Rate"}
        if core_train_display:
            lines.append("-" * width)
            for k, v in core_train_display.items():
                lines.append(f"{k + ':':<24}{v:.4f}")

        lines.append("-" * width)
        print("\n".join(lines), flush=True)

    def _eta(self) -> str:
        if not self.total_timesteps or self.timestep <= 0:
            return "n/a"

        progress = self.timestep / self.total_timesteps
        if progress <= 0:
            return "n/a"

        elapsed = time.time() - self._start_time
        remaining_seconds = max(0, int(elapsed * (1 - progress) / progress))

        hours, remainder = divmod(remaining_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"

    def write_data(self):
        """Flush every tracked tag to tensorboard and/or wandb."""
        if self._wandb is not None:
            grouped: Dict[int, Dict[str, float]] = collections.defaultdict(dict)
            for k, v in self._tracking_data.items():
                grouped[self._tracking_step[k]][k] = v
            for step in sorted(grouped):
                self._wandb.log(grouped[step], step=step)

        if self._tb_writer is not None:
            for k, v in self._tracking_data.items():
                self._tb_writer.add_scalar(tag=k, scalar_value=v, global_step=self._tracking_step[k])

        # reset data containers
        self._tracking_data.clear()
        self._tracking_step.clear()

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a single scalar value."""
        if self._tb_writer is not None:
            self._tb_writer.add_scalar(tag, value, step)
        if self._wandb is not None:
            self._wandb.log({tag: value}, step=step)

    def log_scalars(self, metrics: Dict[str, float], step: int) -> None:
        """Log multiple scalars at once (e.g. reward, loss, kl in one call)."""
        if self._tb_writer is not None:
            for tag, value in metrics.items():
                self._tb_writer.add_scalar(tag, value, step)
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def log_histogram(self, tag: str, values, step: int) -> None:
        """Log a histogram (e.g. action distribution, gradient norms)."""
        if self._tb_writer is not None:
            self._tb_writer.add_histogram(tag, values, step)
        if self._wandb is not None:
            import wandb

            self._wandb.log({tag: wandb.Histogram(values)}, step=step)

    def log_video(self, tag: str, video, step: int, fps: int = 30) -> None:
        """Log a video, e.g. shape (N, T, C, H, W) as expected by tensorboard."""
        if self._tb_writer is not None:
            self._tb_writer.add_video(tag, video, step, fps=fps)
        if self._wandb is not None:
            self._wandb.log({tag: self._wandb.Video(video, fps=fps)}, step=step)

    def close(self) -> None:
        """Flush and close all backends."""
        if self._tb_writer is not None:
            self._tb_writer.flush()
            self._tb_writer.close()
        if self._wandb is not None:
            self._wandb.finish()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
