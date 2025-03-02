import numpy as np
from dataclasses import dataclass

from . import BaseTrajectory, BaseTrajectoryCfg
from . import Registerable, RegisterableCfg


@dataclass
class SquareTrajectoryCfg(BaseTrajectoryCfg, RegisterableCfg):
    x_dim: int = 1.0
    y_dim: int = 1.0


class SquareTrajectory(BaseTrajectory, Registerable):
    _cfg: SquareTrajectoryCfg

    def __init__(self, cfg: SquareTrajectoryCfg) -> None:
        super().__init__(cfg)

    def generate_trajectory(self) -> None:
        x1 = np.linspace(-self._cfg.x_dim / 2, self._cfg.x_dim / 2, num=250)
        y1 = np.ones(250) * self._cfg.y_dim / 2
        theta1 = np.zeros(250)
        x2 = np.ones(250) * self._cfg.x_dim / 2
        y2 = np.linspace(self._cfg.y_dim / 2, -self._cfg.y_dim / 2, num=250)
        theta2 = np.ones(250) * np.pi / 2
        x3 = np.linspace(self._cfg.x_dim / 2, -self._cfg.x_dim / 2, num=250)
        y3 = np.ones(250) * -self._cfg.y_dim / 2
        theta3 = np.ones(250) * np.pi
        x4 = np.ones(250) * -self._cfg.x_dim / 2
        y4 = np.linspace(-self._cfg.y_dim / 2, self._cfg.y_dim / 2, num=250)
        theta4 = np.ones(250) * (-np.pi / 2)
        x = np.concatenate((x1, x2, x3, x4))
        y = np.concatenate((y1, y2, y3, y4))
        theta = np.concatenate((theta1, theta2, theta3, theta4))
        self._trajectory = np.stack((x, y), axis=1)
        self._trajectory_angle = theta
