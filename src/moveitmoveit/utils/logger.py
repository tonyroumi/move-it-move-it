from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional


class Logger:
    """ Simple RL logger with optional TensorBoard and/or Weights & Biases backends. """

    def __init__(
        self,
        backend: str = "tensorboard",
        log_dir: str = "runs",
        project: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        wandb_kwargs: Optional[Dict[str, Any]] = None,
    ):
        assert backend in ("tensorboard", "wandb", "both", "none"), (
            f"Unknown backend: {backend}"
        )
        self.backend = backend
        self.run_name = run_name or time.strftime("run_%Y%m%d_%H%M%S")

        self._tb_writer = None
        self._wandb = None

        use_tb = backend in ("tensorboard", "both")
        use_wandb = backend in ("wandb", "both")

        if use_tb:
            from torch.utils.tensorboard import SummaryWriter

            full_dir = os.path.join(log_dir, self.run_name)
            os.makedirs(full_dir, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=full_dir)

        if use_wandb:
            import wandb

            wandb.init(
                project=project,
                name=self.run_name,
                config=config,
                **(wandb_kwargs or {}),
            )
            self._wandb = wandb

        if config and self._tb_writer is not None:
            # Dump hyperparams as text since SummaryWriter has no native dict logger
            cfg_str = "\n".join(f"{k}: {v}" for k, v in config.items())
            self._tb_writer.add_text("config", cfg_str)

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