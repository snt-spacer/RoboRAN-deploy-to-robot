import numpy as np
from dataclasses import dataclass

from . import BaseTrajectory, BaseTrajectoryCfg
from . import Registerable, RegisterableCfg


@dataclass
class GeronoLemniscateTrajectoryCfg(BaseTrajectoryCfg, RegisterableCfg):
    """Configuration for the Gerono lemniscate trajectory.
    
    More information about the Gerono lemniscate can be found at:
    https://en.wikipedia.org/wiki/Lemniscate_of_Gerono
    """
    a: float = 1.0
    """The parameter a of the Gerono lemniscate. Default is 1.0."""


class GeronoLemniscateTrajectory(BaseTrajectory, Registerable):
    """Gerono lemniscate trajectory generator.

    The trajectory is a Gerono lemniscate. More information about the Gerono lemniscate can be found at
    https://en.wikipedia.org/wiki/Lemniscate_of_Gerono
    """
    _cfg: GeronoLemniscateTrajectoryCfg

    def __init__(self, cfg: GeronoLemniscateTrajectoryCfg) -> None:
        """Initialize the Gerono lemniscate trajectory generator."""
        super().__init__(cfg)

    def generate_trajectory(self) -> None:
        """Generate the Gerono lemniscate trajectory."""
        t = np.linspace(0, 2 * np.pi, num=1000)
        x = self._cfg.a * np.cos(t)
        y = self._cfg.a * np.sin(t) * np.cos(t)
        self._trajectory = np.stack((x, y), axis=1)
