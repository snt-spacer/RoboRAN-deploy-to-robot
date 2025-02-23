from . import Registerable
from . import BaseFormater, BaseFormaterCfg
from fp_inference.state_preprocessors import BaseStatePreProcessor

from geometry_msgs.msg import PointStamped
from dataclasses import dataclass
import numpy as np
import gymnasium
import torch


@dataclass
class GoToPositionTaskCfg(BaseFormaterCfg):
    """Configuration for the go to position task."""

    position_tolerance: float = 0.1
    terminate_early: bool = False


class GoToPositionFormater(Registerable, BaseFormater):
    """Task formatter for the go to position task."""

    _task_cfg: GoToPositionTaskCfg

    def __init__(
        self,
        state_preprocessor: BaseStatePreProcessor | None = None,
        device: str = "cuda",
        max_steps: int = 500,
        task_cfg: GoToPositionTaskCfg = GoToPositionTaskCfg(),
        num_actions: int = 2,
        **kwargs,
    ) -> None:
        """Task formatter for the go to position task."""
        super().__init__(state_preprocessor, device, max_steps, task_cfg)

        # General parameters
        self.ROS_TYPE = PointStamped

        # Task parameters
        self._observation_space = gymnasium.spaces.Box(-np.inf, np.inf, (6 + num_actions,))
        self._task_data = torch.zeros((1, 6), device=self._device)
        self._target_position = torch.zeros((1, 2), device=self._device)

    def build_logs(self) -> None:
        """Build the logs for the task.

        Logs are used by the logger to automatically keep track of the task's progress. This provides a unified way
        to keep track of the task's progress and to compare different tasks and robots.
        """
        super().build_logs()

        # Task logs
        self._logs["distance_error"] = torch.zeros((1, 1), device=self._device)
        self._logs["target_heading_error"] = torch.tensor((1, 1), device=self._device)
        self._logs["target_position"] = torch.tensor((1, 2), device=self._device)
        self._logs["task_data"] = torch.zeros((1, 6), device=self._device)

        # Log specifications
        self._logs_specs["distance_error"] = [".m"]
        self._logs_specs["target_heading_error"] = [".rad"]
        self._logs_specs["target_position"] = [".x.m", ".y.m"]
        self._logs_specs["task_data"] = [
            ".dist.m",
            ".cos(heading).u",
            ".sin(heading).u",
            ".lin_vel_body.x.m/s",
            ".lin_vel_body.y.m/s",
            ".ang_vel_body.rad/s",
        ]

    def update_logs(self) -> None:
        """Update the logs for the task.

        This method is called every time the logs are accessed. The logs should be updated based on the current state
        of the task. The logger performs a deep copy of the logs after this method is called, so the logs can be
        safely modified in place.
        """
        self._logs["distance_error"] = self._dist
        self._logs["target_heading_error"] = self._target_heading_error
        self._logs["target_position"] = self._target_position
        self._logs["task_data"] = self._task_data

    def check_task_completion(self) -> None:
        """Check if the task has been completed."""
        if self._task_cfg.terminate_early:
            cart_dist_bool = self._dist < self._task_cfg.position_tolerance
        else:
            cart_dist_bool = False
        time_bool = self._step >= self._max_steps

        self._task_completed = cart_dist_bool or time_bool

    def format_observation(self, actions: torch.Tensor | None = None) -> None:
        """Format the observation for the task.

        Args:
            actions (torch.Tensor): The actions to be formatted with the observation.
        """
        super().format_observation(actions)
        # Position distance
        self._dist = torch.linalg.norm(
            self._target_position - self._state_preprocessor.position[:, :2], dim=1, keepdim=True
        )
        # Heading distance
        target_heading_w = torch.atan2(
            self._target_position[:, 1] - self._state_preprocessor.position[:, 1],
            self._target_position[:, 0] - self._state_preprocessor.position[:, 0],
        )

        self._target_heading_error = torch.atan2(
            torch.sin(target_heading_w - self._state_preprocessor.heading),
            torch.cos(target_heading_w - self._state_preprocessor.heading),
        )

        self._task_data[:, 0] = self._dist
        self._task_data[:, 1] = torch.cos(self._target_heading_error)
        self._task_data[:, 2] = torch.sin(self._target_heading_error)
        self._task_data[:, 3:5] = self._state_preprocessor.linear_velocities_body[:, :2]
        self._task_data[:, 5] = self._state_preprocessor.angular_velocities_body[:, -1]
        self._observation = torch.cat((self._task_data, actions), dim=1)
        self.check_task_completion()

    def update_goal_ROS(self, position: PointStamped | None = None, **kwargs) -> None:
        """Update the goal position using the ROS message.

        When a goal is received, the task is live and the number of steps is reset.

        Args:
            position (PointStamped): The goal position.
        """
        if position is not None:
            print("Received new goal")
            print("Going to position: ", position.point.x, position.point.y)

            self._step += 1
            self._target_position[0, 0] = position.point.x
            self._target_position[0, 1] = position.point.y
            # A goal has been received the task is live
            self._task_is_live = True
            # Reset the number of steps to 0 when a new goal is received
            self._step = 0

    def reset(self) -> None:
        """Reset the task to its initial state."""
        super().reset()

        self._target_position.fill_(0)
        self._task_data.fill_(0)
