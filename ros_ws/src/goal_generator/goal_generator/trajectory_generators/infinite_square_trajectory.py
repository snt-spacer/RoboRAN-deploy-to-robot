import numpy as np
from dataclasses import dataclass

from . import BaseTrajectory, BaseTrajectoryCfg
from . import Registerable, RegisterableCfg


@dataclass
class InfiniteSquareTrajectoryCfg(BaseTrajectoryCfg, RegisterableCfg):
    x_dim: int = 1.0
    y_dim: int = 1.0


class InfiniteSquareTrajectory(BaseTrajectory, Registerable):
    _cfg: InfiniteSquareTrajectoryCfg

    def __init__(self, cfg: InfiniteSquareTrajectoryCfg) -> None:
        super().__init__(cfg)

    def generate_trajectory(self) -> np.ndarray:
        x1 = np.linspace(0, self._cfg.x_dim, num=250)
        y1 = np.zeros(250)
        theta1 = np.zeros(250)
        x2 = np.ones(250) * self._cfg.x_dim
        y2 = np.linspace(0, self._cfg.y_dim, num=250)
        theta2 = np.ones(250) * np.pi / 2
        x3 = np.linspace(self._cfg.x_dim, 0, num=250)
        y3 = np.ones(250) * self._cfg.y_dim
        theta3 = np.ones(250) * np.pi
        x4 = np.zeros(500)
        y4 = np.linspace(self._cfg.y_dim, -self._cfg.y_dim, num=500)
        theta4 = np.ones(500) * (-np.pi / 2)
        x5 = np.linspace(0, -self._cfg.x_dim, num=250)
        y5 = np.ones(250) * -self._cfg.y_dim
        theta5 = np.ones(250) * np.pi
        x6 = np.ones(250) * -self._cfg.x_dim
        y6 = np.linspace(-self._cfg.y_dim, 0, num=250)
        theta6 = np.ones(250) * np.pi / 2
        x7 = np.linspace(-self._cfg.x_dim, 0, num=250)
        y7 = np.zeros(250)
        theta7 = np.zeros(250)
        x = np.concatenate((x1, x2, x3, x4, x5, x6, x7))
        y = np.concatenate((y1, y2, y3, y4, y5, y6, y7))
        theta = np.concatenate((theta1, theta2, theta3, theta4, theta5, theta6, theta7))
        self._trajectory = np.stack((x, y), axis=1)
        self._trajectory_angle = theta
