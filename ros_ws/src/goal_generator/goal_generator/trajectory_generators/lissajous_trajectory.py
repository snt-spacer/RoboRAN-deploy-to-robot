import numpy as np
from dataclasses import dataclass

from . import BaseTrajectory, BaseTrajectoryCfg
from . import Registerable, RegisterableCfg


@dataclass
class LissajousTrajectoryCfg(BaseTrajectoryCfg, RegisterableCfg):
    A: float = 1.0
    B: float = 1.0
    a: float = 5.0
    b: float = 4.0
    omega_x: float = np.pi/2.0


class LissajousTrajectory(BaseTrajectory, Registerable):
    _cfg: LissajousTrajectoryCfg

    def __init__(self, cfg: LissajousTrajectoryCfg) -> None:
        super().__init__(cfg)

    def generate_trajectory(self) -> None:
        theta = np.linspace(0, 2 * np.pi, num=10000)
        x = self._cfg.A * np.sin(self._cfg.a * theta + self._cfg.omega_x)
        y = self._cfg.B * np.sin(self._cfg.b * theta)
        self._trajectory = np.stack((x, y), axis=1)
        self._trajectory_angle = np.arctan2(y, x)
