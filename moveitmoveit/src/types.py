from abc import ABC
from dataclasses import dataclass

@dataclass(frozen=True)
class BaseParams(ABC):
    @classmethod
    def from_dict(cls, cfg: dict) -> "BaseParams":
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})