from __future__ import annotations

from typing import Any, Dict, Optional
import collections
import os
import time

import numpy as np

class Logger:
    """ Simple RL logger with optional TensorBoard and/or Weights & Biases backends. """

    def __init__(
        self,
        backend: str = "tensorboard",
        log_dir: str = "logs",
        project: Optional[str] = None,
        exp_cfg: Optional[Dict[str, Any]] = None,
        wandb_kwargs: Optional[Dict[str, Any]] = None,
    ):
        assert backend in ("tensorboard", "wandb", "both", "none"), (
            f"Unknown backend: {backend}"
        )
        self.backend = backend
        self.log_dir = log_dir
        self.exp_cfg = exp_cfg

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
                project=project,
                **(wandb_kwargs or {}),
            )
            self._wandb = wandb

        self._tracking_data = collections.defaultdict(list)
        self._tracking_step = collections.defaultdict(int)

    def track_data(self, *, tag: str, value: float, step: int):
        self._tracking_data[tag].append(value)
        self._tracking_step[tag] = step

    def write_tracking_data(self):
        for k, v in self._tracking_data.items():
            if k.endswith("(min)"):
                self.writer.add_scalar(tag=k, value=np.min(v), timestep=self._tracking_step[k])
            elif k.endswith("(max)"):
                self.writer.add_scalar(tag=k, value=np.max(v), timestep=self._tracking_step[k])
            else:
                self.writer.add_scalar(tag=k, value=np.mean(v), timestep=self._tracking_step[k])

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
