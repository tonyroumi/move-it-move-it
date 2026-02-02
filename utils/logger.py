import logging
import os

from torch.utils.tensorboard import SummaryWriter

from .array import ArrayUtils


class Logger:
    def __init__(
        self,
        log_dir: str = "logs",
        verbose: bool = False,
        use_wandb: bool = False,
        wandb_project: str | None = None,
        wandb_run_name: str | None = None,
    ):
        os.makedirs(log_dir, exist_ok=True)

        self._logger = logging.getLogger()
        self._logger.setLevel(logging.INFO)

        self.writer = SummaryWriter(log_dir=log_dir)

        self.verbose = verbose
        self.step_num = 0
        self.epoch_num = 0

        # ---- W&B ----
        self.use_wandb = use_wandb
        self.wandb = None

        if self.use_wandb:
            try:
                import wandb  # pylint: disable=import-outside-toplevel
                self.wandb = wandb
                self.wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    dir=log_dir,
                )
            except ImportError as exc:
                raise RuntimeError("use_wandb=True but wandb is not installed") from exc

    def step(self):
        self.step_num += 1

    def epoch(self):
        self.epoch_num += 1
        if self.use_wandb:
            self.wandb.log({"epoch": self.epoch_num}, step=self.step_num)

    def info(self, msg: str):
        self._logger.info(msg)

    def warn(self, msg: str):
        self._logger.warning(msg)

    def error(self, msg: str):
        self._logger.error(msg)

    def log_metric(self, name: str, value: float):
        if ArrayUtils.is_tensor(value):
            value = value.detach().item()

        if self.verbose:
            self.info(f"{name}: {value} (step={self.step_num})")

        # TensorBoard
        self.writer.add_scalar(name, value, self.step_num)

        # W&B
        if self.use_wandb:
            self.wandb.log({name: value}, step=self.step_num)

    def close(self):
        self.writer.close()
        if self.use_wandb:
            self.wandb.finish()
