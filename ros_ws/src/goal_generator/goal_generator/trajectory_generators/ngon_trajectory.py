import numpy as np
from dataclasses import dataclass

from . import BaseTrajectory, BaseTrajectoryCfg
from . import Registerable, RegisterableCfg

@dataclass
class NGonTrajectoryCfg(BaseTrajectoryCfg, RegisterableCfg):
    num_sides: int = 3
    size: float = 1.0

    def __post_init__(self):
        assert self.num_sides >= 3, "Number of sides must be at least 3"

class NGonTrajectory(BaseTrajectory, Registerable):
    _cfg: NGonTrajectoryCfg
    def __init__(self, cfg: NGonTrajectoryCfg) -> None:
        super().__init__(cfg)

    def generate_trajectory(self) -> None:
        num_points = int(1000 / self._cfg.num_sides)
        t = np.linspace(0, 1, num=num_points)
        x = []
        y = []
        for i in range(0, self._cfg.num_sides):
            x.append((1-t)*np.cos(2*i*np.pi/self._cfg.num_sides) + t*np.cos(2*(i+1)*np.pi/self._cfg.num_sides))
            y.append((1-t)*np.sin(2*i*np.pi/self._cfg.num_sides) + t*np.sin(2*(i+1)*np.pi/self._cfg.num_sides))
        x = self._cfg.size * np.concatenate(x) / 2.0
        y = self._cfg.size * np.concatenate(y) / 2.0

        self._trajectory = np.stack((x, y), axis=1)
        self._trajectory_angle = np.arctan2(y, x) 