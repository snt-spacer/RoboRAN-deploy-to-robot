import numpy as np
from dataclasses import dataclass

from . import BaseTrajectory, BaseTrajectoryCfg
from . import Registerable, RegisterableCfg


@dataclass
class SinewaveTrajectoryCfg(BaseTrajectoryCfg, RegisterableCfg):
    amplitude: float = 1.0
    frequency: float = 1.0
    num_periods: int = 5


class SinewaveTrajectory(BaseTrajectory, Registerable):
    _cfg: SinewaveTrajectoryCfg

    def __init__(self, cfg: SinewaveTrajectoryCfg) -> None:
        super().__init__(cfg)

    def generate_trajectory(self) -> None:
        t = np.linspace(0, 2 * np.pi * self._cfg.num_periods / self._cfg.frequency, num=1000)
        x = t
        y = self._cfg.amplitude * np.sin(self._cfg.frequency * t)

        # Circle
        self._trajectory = np.stack((x, y), axis=1)
        # Tangent angle
        self._trajectory_angle = np.arctan2(y, x)
