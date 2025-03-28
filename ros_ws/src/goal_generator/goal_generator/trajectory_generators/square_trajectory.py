import numpy as np
from dataclasses import dataclass

from . import BaseTrajectory, BaseTrajectoryCfg
from . import Registerable, RegisterableCfg


@dataclass
class SquareTrajectoryCfg(BaseTrajectoryCfg, RegisterableCfg):
    """Configuration for the square trajectory."""
    x_dim: int = 1.0
    """The x dimension of the square. Default is 1.0."""
    y_dim: int = 1.0
    """The y dimension of the square. Default is 1.0."""


class SquareTrajectory(BaseTrajectory, Registerable):
    """Square trajectory generator."""
    _cfg: SquareTrajectoryCfg

    def __init__(self, cfg: SquareTrajectoryCfg) -> None:
        """Initialize the square trajectory generator."""
        super().__init__(cfg)

    def generate_trajectory(self) -> None:
        """Generate the square trajectory."""
        x1 = np.linspace(-self._cfg.x_dim / 2, self._cfg.x_dim / 2, num=250)
        y1 = np.ones(250) * self._cfg.y_dim / 2
        x2 = np.ones(250) * self._cfg.x_dim / 2
        y2 = np.linspace(self._cfg.y_dim / 2, -self._cfg.y_dim / 2, num=250)
        x3 = np.linspace(self._cfg.x_dim / 2, -self._cfg.x_dim / 2, num=250)
        y3 = np.ones(250) * -self._cfg.y_dim / 2
        x4 = np.ones(250) * -self._cfg.x_dim / 2
        y4 = np.linspace(-self._cfg.y_dim / 2, self._cfg.y_dim / 2, num=250)
        x = np.concatenate((x1, x2, x3, x4))
        y = np.concatenate((y1, y2, y3, y4))
        self._trajectory = np.stack((x, y), axis=1)
