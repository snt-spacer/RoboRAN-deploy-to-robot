import numpy as np
from scipy.interpolate import interp1d
from dataclasses import dataclass


@dataclass
class BaseTrajectoryCfg:
    point_distance: float = 0.05
    position_offset: tuple[float, float] = (0.0, 0.0)
    angle_offset: float = 0.0


class BaseTrajectory:
    def __init__(self, cfg: BaseTrajectoryCfg) -> None:
        self._cfg = cfg
        self._trajectory = None
        self._trajectory_angle = None

    @property
    def trajectory(self) -> tuple[np.ndarray, np.ndarray]:
        if (self._trajectory is None) or (self._trajectory_angle is None):
            self.generate_trajectory()
            self.apply_trajectory_offset()
            # self.reparametrize()
        return self._trajectory, self._trajectory_angle

    def generate_trajectory(self) -> np.ndarray:
        raise NotImplementedError

    def apply_trajectory_offset(self) -> np.ndarray:
        # Apply rotation then translation
        x, y = self._trajectory[:, 0], self._trajectory[:, 1]
        x_rot = x * np.cos(self._cfg.angle_offset) - y * np.sin(self._cfg.angle_offset)
        y_rot = x * np.sin(self._cfg.angle_offset) + y * np.cos(self._cfg.angle_offset)
        self._trajectory_angle += self._cfg.angle_offset
        x_rot += self._cfg.position_offset[0]
        y_rot += self._cfg.position_offset[1]
        self._trajectory = np.stack((x_rot, y_rot), axis=1)

    def reparametrize(self) -> np.ndarray:
        s = np.cumsum(np.sqrt(np.sum(np.square(np.diff(self._trajectory, axis=0)), axis=1)))
        interp = interp1d(s, self._trajectory, kind="linear")
        num_points = s[-1] / self._cfg.point_distance
        s_new = np.linspace(0, s[-1], num=num_points)
        self._trajectory = interp(s_new)
        interp_angle = interp1d(s, self._trajectory_angle, kind="linear")
        self._trajectory_angle = interp_angle(s_new)
