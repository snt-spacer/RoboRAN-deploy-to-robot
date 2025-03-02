import numpy as np
from dataclasses import dataclass

from . import BaseTrajectory, BaseTrajectoryCfg
from . import Registerable, RegisterableCfg


@dataclass
class SpiralTrajectoryCfg(BaseTrajectoryCfg, RegisterableCfg):
    min_radius: float = 0.25
    max_radius: float = 2.0
    num_turns: float = 5.0


class SpiralTrajectory(BaseTrajectory, Registerable):
    _cfg: SpiralTrajectoryCfg

    def __init__(self, cfg: SpiralTrajectoryCfg) -> None:
        super().__init__(cfg)

    def generate_trajectory(self) -> None:
        num_points = 10000
        t = np.linspace(0, 2 * np.pi * self._cfg.num_turns, num=num_points)
        r = np.linspace(self._cfg.min_radius, self._cfg.max_radius, num=num_points)
        x = r * np.cos(t)
        y = r * np.sin(t)

        # Spiral
        self._trajectory = np.stack((x, y), axis=1)
        # Tangent angle
        self._trajectory_angle = np.arctan2(y, x)
