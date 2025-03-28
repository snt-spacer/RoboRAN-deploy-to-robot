import numpy as np
from dataclasses import dataclass

from . import BaseTrajectory, BaseTrajectoryCfg
from . import Registerable, RegisterableCfg


@dataclass
class AcceleratingSinewaveTrajectoryCfg(BaseTrajectoryCfg, RegisterableCfg):
    """Configuration for the accelerating sinewave trajectory.
    
    The trajectory is a sinewave with an increasing frequency and decreasing amplitude.
    """
    min_frequency: float = 1.0
    """The minimum frequency of the sinewave. Default is 1.0."""
    max_frequency: float = 3.0
    """The maximum frequency of the sinewave. Default is 3.0."""
    num_periods: int = 5
    """The number of periods of the sinewave. Default is 5."""
    max_amplitude: float = 3.0
    """The maximum amplitude of the sinewave. Default is 3.0."""
    min_amplitude: float = 1.0
    """The minimum amplitude of the sinewave. Default is 1.0."""


class AcceleratingSinewaveTrajectory(BaseTrajectory, Registerable):
    """Accelerating sinewave trajectory generator.
    
    The trajectory is a sinewave with an increasing frequency and decreasing amplitude.
    """
    _cfg: AcceleratingSinewaveTrajectoryCfg

    def __init__(self, cfg: AcceleratingSinewaveTrajectoryCfg) -> None:
        """Initialize the accelerating sinewave trajectory generator."""
        super().__init__(cfg)

    def generate_trajectory(self) -> None:
        """Generate the accelerating sinewave trajectory.

        The trajectory is a sinewave with an increasing frequency and decreasing amplitude.
        Both the frequency and amplitude are linearly interpolated between the minimum and maximum values.
        """
        t = np.linspace(0, 1, num=1000)
        x = t * 2 * np.pi * self._cfg.num_periods / ((self._cfg.max_frequency + self._cfg.min_frequency) / 2)
        amplitude = self._cfg.min_amplitude + (self._cfg.max_amplitude - self._cfg.min_amplitude) * (1 - t)
        frequency = self._cfg.min_frequency + (self._cfg.max_frequency - self._cfg.min_frequency) * t
        y = amplitude * np.sin(x * frequency)
        self._trajectory = np.stack((x, y), axis=1)
