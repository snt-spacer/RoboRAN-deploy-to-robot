import numpy as np
from dataclasses import dataclass

from . import BaseTrajectory, BaseTrajectoryCfg
from . import Registerable, RegisterableCfg


@dataclass
class HippopedeTrajectoryCfg(BaseTrajectoryCfg, RegisterableCfg):
    """Configuration for the hippopede trajectory.
    
    More information about the hippopede can be found at:
    https://en.wikipedia.org/wiki/Hippopede
    """
    a: float = 1.0
    """The parameter a of the hippopede. Default is 1.0."""
    b: float = 1.0
    """The parameter b of the hippopede. Default is 1.0."""


class HippopedeTrajectory(BaseTrajectory, Registerable):
    """Hippopede trajectory generator.
    
    The trajectory is a hippopede. More information about the hippopede can be found at
    https://en.wikipedia.org/wiki/Hippopede
    """
    _cfg: HippopedeTrajectoryCfg

    def __init__(self, cfg: HippopedeTrajectoryCfg) -> None:
        """Initialize the hippopede trajectory generator."""
        super().__init__(cfg)

    def generate_trajectory(self) -> None:
        """Generate the hippopede trajectory."""
        theta = np.linspace(0, 2 * np.pi, num=1000)
        r = 4 * self._cfg.b * (self._cfg.a - self._cfg.b * np.sin(theta) ** 2)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        self._trajectory = np.stack((x, y), axis=1)
