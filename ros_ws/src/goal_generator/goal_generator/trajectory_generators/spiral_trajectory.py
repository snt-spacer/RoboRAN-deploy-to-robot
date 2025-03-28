import numpy as np
from dataclasses import dataclass

from . import BaseTrajectory, BaseTrajectoryCfg
from . import Registerable, RegisterableCfg


@dataclass
class SpiralTrajectoryCfg(BaseTrajectoryCfg, RegisterableCfg):
    """Configuration for the spiral trajectory.
    
    The trajectory is a spiral with a varying radius from min_radius to max_radius over num_turns.
    """
    min_radius: float = 0.25
    """The minimum radius of the spiral. Default is 0.25."""
    max_radius: float = 2.0
    """The maximum radius of the spiral. Default is 2.0."""
    num_turns: float = 5.0
    """The number of turns of the spiral. Default is 5.0."""


class SpiralTrajectory(BaseTrajectory, Registerable):
    """Spiral trajectory generator.

    The trajectory is a spiral with a varying radius from min_radius to max_radius over num_turns.
    """
    _cfg: SpiralTrajectoryCfg

    def __init__(self, cfg: SpiralTrajectoryCfg) -> None:
        """Initialize the spiral trajectory generator."""
        super().__init__(cfg)

    def generate_trajectory(self) -> None:
        """Generate the spiral trajectory."""
        num_points = 10000
        t = np.linspace(0, 2 * np.pi * self._cfg.num_turns, num=num_points)
        r = np.linspace(self._cfg.min_radius, self._cfg.max_radius, num=num_points)
        x = r * np.cos(t)
        y = r * np.sin(t)
        self._trajectory = np.stack((x, y), axis=1)
