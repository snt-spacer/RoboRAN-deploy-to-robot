from . import Registerable
from . import BaseFormater
from fp_inference.state_preprocessors import BaseStatePreProcessor

from geometry_msgs.msg import PointStamped
import torch


class GoToPositionFormater(Registerable, BaseFormater):
    def __init__(self, state_preprocessor: BaseStatePreProcessor | None = None, device: str = 'cuda', **kwargs):
        super().__init__(state_preprocessor, device)

        # General parameters
        self.ROS_TYPE = PointStamped

        self._task_data = torch.zeros((1, 6), device=self._device)
        self._target_position = torch.zeros((1, 2), device=self._device)

    def build_logs(self):
        super().build_logs()
        self._logs["distance_error"] = 0
        self._logs["heading_error"] = 0

    def update_logs(self):
        self._logs["distance_error"] = self.dist
        self._logs["position_heading_error"] = self.target_heading_error

    def get_observation(self, actions: torch.Tensor | None = None) -> torch.Tensor | None:
        if actions is None:
            return None

        # Position distance
        self.dist = torch.linalg.norm(self._target_position - self._state_preprocessor.position[:, :2], dim=1)
        # Heading distance
        target_heading_w = torch.atan2(
            self._target_position[:, 1] - self._state_preprocessor.position[:, 1],
            self._target_position[:, 0] - self._state_preprocessor.position[:, 0],
        )
        self.target_heading_error = torch.atan2(torch.sin(target_heading_w - self._state_preprocessor.heading), torch.cos(target_heading_w - self._state_preprocessor.heading))
        self._task_data[:, 0]

        # Store in buffer [distance, cos(angle), sin(angle), lin_vel_x, lin_vel_y, ang_vel, prev_action]
        self._task_data[:, 0] = self.dist
        self._task_data[:, 1] = torch.cos(self.target_heading_error)
        self._task_data[:, 2] = torch.sin(self.target_heading_error)
        self._task_data[:, 3:5] = self._state_preprocessor.linear_velocities_body[:, :2]
        self._task_data[:, 5] = self._state_preprocessor.angular_velocities_body[:, -1]

        return torch.cat((self._task_data, actions), dim=1)
    
    def update_goal(self, position: torch.Tensor | None = None, **kwargs):
        if position is not None:
            self._step += 1
            self._target_position[0] = position[:, :2]
    
    def update_goal_ROS(self, position: PointStamped | None = None, **kwargs):
        if position is not None:
            self._step += 1
            self._target_position[0, 0] = position.point.x
            self._target_position[0, 1] = position.point.y

    def reset(self):
        self._step = 0
        self._last_preprocessor_step = 0
        self._observation_step = 0

