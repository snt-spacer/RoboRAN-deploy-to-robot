import numpy as np
from dataclasses import dataclass

from . import BaseTrajectory, BaseTrajectoryCfg
from . import Registerable, RegisterableCfg

@dataclass
class BernouilliLemniscateTrajectoryCfg(BaseTrajectoryCfg, RegisterableCfg):
    a: float = 1.0

class BernouilliLemniscateTrajectory(BaseTrajectory, Registerable):
    _cfg: BernouilliLemniscateTrajectoryCfg
    def __init__(self, cfg: BernouilliLemniscateTrajectoryCfg) -> None:
        super().__init__(cfg)

    def generate_trajectory(self) -> None:
        t = np.linspace(0, 2 * np.pi, num=1000)
        x = self._cfg.a * np.cos(t) / (1 + np.sin(t)**2)
        y = self._cfg.a * np.sin(t) * np.cos(t) / (1 + np.sin(t)**2) 
        self._trajectory = np.stack((x, y), axis=1)
        self._trajectory_angle = np.arctan2(y, x)
        
