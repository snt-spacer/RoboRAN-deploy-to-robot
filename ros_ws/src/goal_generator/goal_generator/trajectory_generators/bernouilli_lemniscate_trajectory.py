import numpy as np
from dataclasses import dataclass

from . import BaseTrajectory, BaseTrajectoryCfg
from . import Registerable, RegisterableCfg


@dataclass
class BernouilliLemniscateTrajectoryCfg(BaseTrajectoryCfg, RegisterableCfg):
    """Configuration for the Bernouilli lemniscate trajectory.
    
    More information about the Bernouilli lemniscate can be found at:
    https://en.wikipedia.org/wiki/Lemniscate_of_Bernoulli
    """
    a: float = 1.0
    """The parameter a of the Bernouilli lemniscate. Default is 1.0."""


class BernouilliLemniscateTrajectory(BaseTrajectory, Registerable):
    """Bernouilli lemniscate trajectory generator.

    The trajectory is a Bernouilli lemniscate. More information about the Bernouilli lemniscate can be found at
    https://en.wikipedia.org/wiki/Lemniscate_of_Bernoulli
    """
    _cfg: BernouilliLemniscateTrajectoryCfg

    def __init__(self, cfg: BernouilliLemniscateTrajectoryCfg) -> None:
        """Initialize the Bernouilli lemniscate trajectory generator."""
        super().__init__(cfg)

    def generate_trajectory(self) -> None:
        """Generate the Bernouilli lemniscate trajectory."""
        t = np.linspace(0, 2 * np.pi, num=1000)
        x = self._cfg.a * np.cos(t) / (1 + np.sin(t) ** 2)
        y = self._cfg.a * np.sin(t) * np.cos(t) / (1 + np.sin(t) ** 2)
        self._trajectory = np.stack((x, y), axis=1)
