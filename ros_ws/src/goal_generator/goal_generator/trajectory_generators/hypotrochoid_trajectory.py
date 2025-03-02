import numpy as np
from dataclasses import dataclass

from . import BaseTrajectory, BaseTrajectoryCfg
from . import Registerable, RegisterableCfg


@dataclass
class HypotrochoidTrajectoryCfg(BaseTrajectoryCfg, RegisterableCfg):
    R: int = 7
    r: int = 4
    d: float = 1.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        assert self.R > self.r, "R must be greater than r"
        assert self.r > self.d, "r must be greater than d"
        assert isinstance(self.R, int), "R must be an integer"
        assert isinstance(self.r, int), "r must be an integer"
        

class HypotrochoidTrajectory(BaseTrajectory, Registerable):
    _cfg: HypotrochoidTrajectoryCfg

    def __init__(self, cfg: HypotrochoidTrajectoryCfg) -> None:
        super().__init__(cfg)

    def generate_trajectory(self) -> None:
        theta = np.linspace(0, 1, num=3000) * np.pi * 2 * np.lcm(self._cfg.R, self._cfg.r) / self._cfg.R
        x = (self._cfg.R - self._cfg.r) * np.cos(theta) + self._cfg.d * np.cos((self._cfg.R - self._cfg.r) / self._cfg.r * theta)
        y = (self._cfg.R - self._cfg.r) * np.sin(theta) - self._cfg.d * np.sin((self._cfg.R - self._cfg.r) / self._cfg.r * theta)
        self._trajectory = np.stack((x, y), axis=1)
        self._trajectory = self._trajectory / np.max(self._trajectory, axis=(0,1)) * self._cfg.scale
        self._trajectory_angle = np.arctan2(y, x)