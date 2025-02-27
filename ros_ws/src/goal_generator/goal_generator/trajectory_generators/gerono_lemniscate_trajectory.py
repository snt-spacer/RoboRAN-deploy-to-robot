import numpy as np
from dataclasses import dataclass

from . import BaseTrajectory, BaseTrajectoryCfg
from . import Registerable, RegisterableCfg

@dataclass
class GeronoLemniscateTrajectoryCfg(BaseTrajectoryCfg, RegisterableCfg):
    a: float = 1.0

class GeronoLemniscateTrajectory(BaseTrajectory, Registerable):
    _cfg: GeronoLemniscateTrajectoryCfg
    def __init__(self, cfg: GeronoLemniscateTrajectoryCfg) -> None:
        super().__init__(cfg)

    def generate_trajectory(self) -> None:
        t = np.linspace(0, 2 * np.pi, num=1000)
        x = self._cfg.a * np.cos(t)
        y = self._cfg.a * np.sin(t) * np.cos(t)
        
        self._trajectory = np.stack((x, y), axis=1)
        self._trajectory_angle = np.arctan2(y, x)
        
