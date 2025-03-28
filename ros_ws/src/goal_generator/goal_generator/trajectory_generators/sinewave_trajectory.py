import numpy as np
from dataclasses import dataclass

from . import BaseTrajectory, BaseTrajectoryCfg
from . import Registerable, RegisterableCfg


@dataclass
class SinewaveTrajectoryCfg(BaseTrajectoryCfg, RegisterableCfg):
    """Configuration for the Sinewave trajectory."""
    amplitude: float = 1.0
    """The amplitude of the sinewave. Default is 1.0."""
    frequency: float = 1.0
    """The frequency of the sinewave. Default is 1.0."""
    num_periods: int = 5
    """The number of periods of the sinewave. Default is 5."""


class SinewaveTrajectory(BaseTrajectory, Registerable):
    """Sinewave trajectory generator."""
    _cfg: SinewaveTrajectoryCfg

    def __init__(self, cfg: SinewaveTrajectoryCfg) -> None:
        """Initialize the Sinewave trajectory generator."""
        super().__init__(cfg)

    def generate_trajectory(self) -> None:
        """Generate the Sinewave trajectory."""
        t = np.linspace(0, 2 * np.pi * self._cfg.num_periods / self._cfg.frequency, num=1000)
        x = t
        y = self._cfg.amplitude * np.sin(self._cfg.frequency * t)
        self._trajectory = np.stack((x, y), axis=1)
