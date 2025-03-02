import numpy as np
from dataclasses import dataclass

from . import BaseTrajectory, BaseTrajectoryCfg
from . import Registerable, RegisterableCfg


@dataclass
class LissajousTrajectoryCfg(BaseTrajectoryCfg, RegisterableCfg):
    """Configuration for the Lissajous trajectory.

    More information about the Lissajous curve can be found at:
    https://en.wikipedia.org/wiki/Lissajous_curve
    """
    A: float = 1.0
    """The scaling factor A (along the x axis) of the Lissajous curve. Default is 1.0."""
    B: float = 1.0
    """The scaling factor B (along the y axis) of the Lissajous curve. Default is 1.0."""
    a: float = 5.0
    """The angular frequency a of the Lissajous curve. Default is 5.0."""
    b: float = 4.0
    """The angular frequency b of the Lissajous curve. Default is 4.0."""
    omega_x: float = np.pi/2.0
    """The phase shift omega_x of the Lissajous curve along the x axis. Default is pi/2.0."""


class LissajousTrajectory(BaseTrajectory, Registerable):
    """Lissajous trajectory generator.

    The trajectory is a Lissajous curve. More information about the Lissajous curve can be found at:
    https://en.wikipedia.org/wiki/Lissajous_curve
    """
    _cfg: LissajousTrajectoryCfg

    def __init__(self, cfg: LissajousTrajectoryCfg) -> None:
        """Initialize the Lissajous trajectory generator."""
        super().__init__(cfg)

    def generate_trajectory(self) -> None:
        """Generate the Lissajous trajectory."""
        theta = np.linspace(0, 2 * np.pi, num=10000)
        x = self._cfg.A * np.sin(self._cfg.a * theta + self._cfg.omega_x)
        y = self._cfg.B * np.sin(self._cfg.b * theta)
        self._trajectory = np.stack((x, y), axis=1)
