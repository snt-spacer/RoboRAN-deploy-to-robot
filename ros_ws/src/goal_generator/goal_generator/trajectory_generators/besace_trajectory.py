
import numpy as np
from dataclasses import dataclass

from . import BaseTrajectory, BaseTrajectoryCfg
from . import Registerable, RegisterableCfg


@dataclass
class BesaceTrajectoryCfg(BaseTrajectoryCfg, RegisterableCfg):
    """Configuration for the Besace trajectory.
    
    More information about the Besace curve can be found at:
    https://fr.wikipedia.org/wiki/Besace_(math%C3%A9matiques)
    """
    a: float = 2.0
    """The parameter a of the Besace curve. Default is 2.0."""
    b: float = 1.0
    """The parameter b of the Besace curve. Default is 1.0."""

class BesaceTrajectory(BaseTrajectory, Registerable):
    """Besace trajectory generator.

    The trajectory is a Besace curve. More information about the Besace curve can be found at
    https://fr.wikipedia.org/wiki/Besace_(math%C3%A9matiques)
    """
    _cfg: BesaceTrajectoryCfg

    def __init__(self, cfg: BesaceTrajectoryCfg) -> None:
        """Initialize the Besace trajectory generator."""
        super().__init__(cfg)

    def generate_trajectory(self) -> None:
        """Generate the Besace trajectory."""
        t = np.linspace(0, 2 * np.pi, num=1000)
        x = self._cfg.a * np.cos(t) - self._cfg.b * np.sin(t)
        y = - np.sin(t) * x
        self._trajectory = np.stack((x, y), axis=1)