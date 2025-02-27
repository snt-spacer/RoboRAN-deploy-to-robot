import numpy as np
from dataclasses import dataclass

from . import BaseTrajectory, BaseTrajectoryCfg
from . import Registerable, RegisterableCfg

@dataclass
class CircleTrajectoryCfg(BaseTrajectoryCfg, RegisterableCfg):
    distance_between_point: float = 0.05
    radius: float = 1.0

class CircleTrajectory(BaseTrajectory, Registerable):
    def __init__(self, cfg: CircleTrajectoryCfg) -> None:
        super().__init__(cfg)

    def generate_trajectory(self) -> np.ndarray:
        t = np.linspace(0, 2 * np.pi, num=1000)
        x = self._cfg.radius * np.cos(t)
        y = self._cfg.radius * np.sin(t)
        
        # Circle
        self._trajectory = np.stack((x, y), axis=1)
        # Tangent angle
        self._trajectory_angle = np.arctan2(y, x)