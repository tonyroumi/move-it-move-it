import logging
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class Logger:
    def __init__(self, log_dir: str = "./logs"):
        os.makedirs(log_dir, exist_ok=True)

        self._logger = logging.getLogger()
        file_handler = logging.FileHandler(os.path.join(log_dir, "loggerOutput.txt"), mode="w")
        file_handler.setFormatter(
            logging.Formatter("[%(levelname)s] %(message)s")
        )
        self._logger.addHandler(file_handler)
        
        self.writer = SummaryWriter(log_dir=log_dir)
    
    def info(self, msg: str):
        self._logger.info(msg)
    
    def warn(self, msg: str):
        self._logger.warning(msg)

    def error(self, msg: str):
        self._logger.error(msg)
    
    def log_metric(self, name: str, value: float, step: int = None):
        self.info(f"{name}, {value}, step: {step}")
        self.writer.add_scalar(name, value, step)
    
    def close(self):
        self.writer.close()