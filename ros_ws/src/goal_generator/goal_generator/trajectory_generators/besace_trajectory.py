
import numpy as np
from dataclasses import dataclass

from . import BaseTrajectory, BaseTrajectoryCfg
from . import Registerable, RegisterableCfg


@dataclass
class BesaceTrajectoryCfg(BaseTrajectoryCfg, RegisterableCfg):
    a: float = 2.0
    b: float = 1.0

class BesaceTrajectory(BaseTrajectory, Registerable):
    _cfg: BesaceTrajectoryCfg

    def __init__(self, cfg: BesaceTrajectoryCfg) -> None:
        super().__init__(cfg)

    def generate_trajectory(self) -> None:
        t = np.linspace(0, 2 * np.pi, num=1000)
        x = self._cfg.a * np.cos(t) - self._cfg.b * np.sin(t)
        y = - np.sin(t) * x
        self._trajectory = np.stack((x, y), axis=1)
        # Tangent angle
        self._trajectory_angle = np.arctan2(y, x)
