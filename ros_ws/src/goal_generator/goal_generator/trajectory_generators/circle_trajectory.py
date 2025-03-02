import numpy as np
from dataclasses import dataclass

from . import BaseTrajectory, BaseTrajectoryCfg
from . import Registerable, RegisterableCfg


@dataclass
class CircleTrajectoryCfg(BaseTrajectoryCfg, RegisterableCfg):
    """Configuration for the circle trajectory."""
    radius: float = 1.0
    """The radius of the circle. Default is 1.0."""


class CircleTrajectory(BaseTrajectory, Registerable):
    """Circle trajectory generator.

    The trajectory is a circle.
    """
    _cfg: CircleTrajectoryCfg

    def __init__(self, cfg: CircleTrajectoryCfg) -> None:
        """Initialize the circle trajectory generator."""
        super().__init__(cfg)

    def generate_trajectory(self) -> None:
        """Generate the circle trajectory."""
        t = np.linspace(0, 2 * np.pi, num=1000)
        x = self._cfg.radius * np.cos(t)
        y = self._cfg.radius * np.sin(t)
        self._trajectory = np.stack((x, y), axis=1)
