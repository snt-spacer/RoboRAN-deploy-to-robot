import numpy as np
from dataclasses import dataclass

from . import BaseTrajectory, BaseTrajectoryCfg
from . import Registerable, RegisterableCfg

@dataclass
class AcceleratingSinewaveTrajectoryCfg(BaseTrajectoryCfg, RegisterableCfg):
    amplitude: float = 1.0
    frequency: float = 1.0
    num_periods: int = 5
    acceleration_rate: float = 1.2
    max_amplitude: float = 2.0
    min_amplitude: float = 1.0


class AcceleratingSinewaveTrajectory(BaseTrajectory, Registerable):
    _cfg: AcceleratingSinewaveTrajectoryCfg
    def __init__(self, cfg: AcceleratingSinewaveTrajectoryCfg) -> None:
        super().__init__(cfg)

    def generate_trajectory(self) -> None:
        t = np.linspace(0, 1, num=1000)
        x = t *2 * np.pi * self._cfg.num_periods / self._cfg.frequency
        amplitude = self._cfg.min_amplitude + (self._cfg.max_amplitude - self._cfg.min_amplitude) * (1 - t)
        y = amplitude * np.sin(self._cfg.frequency * t ** self._cfg.acceleration_rate)
        
        # Circle
        self._trajectory = np.stack((x, y), axis=1)
        # Tangent angle
        self._trajectory_angle = np.arctan2(y, x)