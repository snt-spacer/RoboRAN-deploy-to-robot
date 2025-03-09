import numpy as np
from dataclasses import dataclass

from . import BaseTrajectory, BaseTrajectoryCfg
from . import Registerable, RegisterableCfg


@dataclass
class NGonTrajectoryCfg(BaseTrajectoryCfg, RegisterableCfg):
    """Configuration for the NGon trajectory.
    
    The NGon trajectory is a polygon with n sides of equal length. The minimum number of sides is 3.
    """
    num_sides: int = 3
    """The number of sides of the NGon. Default is 3."""
    size: float = 1.0
    """The size of the NGon. Default is 1.0."""

    def __post_init__(self):
        assert self.num_sides >= 3, "Number of sides must be at least 3"


class NGonTrajectory(BaseTrajectory, Registerable):
    """NGon trajectory generator.

    The trajectory is a polygon with n sides of equal length. The minimum number of sides is 3.
    """
    _cfg: NGonTrajectoryCfg

    def __init__(self, cfg: NGonTrajectoryCfg) -> None:
        """Initialize the NGon trajectory generator."""
        super().__init__(cfg)

    def generate_trajectory(self) -> None:
        """Generate the NGon trajectory."""
        num_points = int(1000 / self._cfg.num_sides)
        t = np.linspace(0, 1, num=num_points)
        x = []
        y = []
        for i in range(0, self._cfg.num_sides):
            x.append(
                (1 - t) * np.cos(2 * i * np.pi / self._cfg.num_sides)
                + t * np.cos(2 * (i + 1) * np.pi / self._cfg.num_sides)
            )
            y.append(
                (1 - t) * np.sin(2 * i * np.pi / self._cfg.num_sides)
                + t * np.sin(2 * (i + 1) * np.pi / self._cfg.num_sides)
            )
        x = self._cfg.size * np.concatenate(x) / 2.0
        y = self._cfg.size * np.concatenate(y) / 2.0
        self._trajectory = np.stack((x, y), axis=1)
