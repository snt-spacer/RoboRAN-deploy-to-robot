from . import Registerable
from . import BaseFormater, BaseFormaterCfg
from rl_inference.state_preprocessors import BaseStatePreProcessor

from geometry_msgs.msg import PoseStamped
from dataclasses import dataclass
from typing import Any
import numpy as np
import gymnasium
import torch


@dataclass
class GoToPoseTaskCfg(BaseFormaterCfg):
    position_tolerance: float = 0.1
    heading_tolerance: float = 0.1
    terminate_early: bool = False


class GoToPoseFormater(Registerable, BaseFormater):
    _task_cfg: GoToPoseTaskCfg

    def __init__(
        self,
        state_preprocessor: BaseStatePreProcessor | None = None,
        device: str = "cuda",
        max_steps: int = 500,
        num_actions: int = 2,
        task_cfg: GoToPoseTaskCfg = GoToPoseTaskCfg(),
        **kwargs,
    ) -> None:

        """Task formatter for the go to pose task. The task is to go to a pose."""
        super().__init__(state_preprocessor, device, max_steps, task_cfg)

        # General parameters
        self.ROS_TYPE = PoseStamped

        # Task parameters
        self._observation_space = gymnasium.spaces.Box(-np.inf, np.inf, (8 + num_actions,))
        self._task_data = torch.zeros((1, 8), device=self._device)
        self._target_position = torch.zeros((1, 2), device=self._device)
        self._target_heading = torch.zeros((1, 1), device=self._device)

    def build_logs(self) -> None:
        """Build the logs for the task.
        
        Logs are used by the logger to automatically keep track of the task's progress. This provides a unified way
        to keep track of the task's progress and to compare different tasks and robots.
        """
        super().build_logs()

        # Task logs
        self._logs["distance_error"] = torch.zeros((1, 1), device=self._device)
        self._logs["target_heading_error"] = torch.zeros((1, 1), device=self._device)
        self._logs["heading_error"] = torch.zeros((1, 1), device=self._device)
        self._logs["target_position"] = torch.zeros((1, 2), device=self._device)
        self._logs["target_heading"] = torch.zeros((1, 1), device=self._device)
        self._logs["task_data"] = torch.zeros((1, 6), device=self._device)

        # Log specifications
        self._logs_specs["distance_error"] = [".m"]
        self._logs_specs["target_heading_error"] = [".rad"]
        self._logs_specs["heading_error"] = [".rad"]
        self._logs_specs["target_position"] = [".x.m", ".y.m"]
        self._logs_specs["target_heading"] = [".rad"]
        self._logs_specs["task_data"] = [
            ".dist.m",
            ".cos(heading_to_target_error).u",
            ".sin(heading_to_target_error).u",
            ".cos(heading_error).u",
            ".sin(heading_error).u",
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
        self._logs["heading_error"] = self._heading_error
        self._logs["target_position"] = self._target_position
        self._logs["target_heading"] = self._target_heading
        self._logs["task_data"] = self._task_data

    def check_task_completion(self) -> None:
        """Check if the task has been completed."""
        if self._task_cfg.terminate_early:
            cart_dist_bool = self._dist < self._task_cfg.position_tolerance
            ang_dist_bool = torch.abs(self._target_heading_error) < self._task_cfg.heading_tolerance
        else:
            cart_dist_bool = False
            ang_dist_bool = False
        time_bool = self._step >= self._max_steps

        self._task_completed = (cart_dist_bool and ang_dist_bool) or time_bool

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
        # Heading error (to the target heading)
        self._heading_error = torch.atan2(
            torch.sin(self._target_heading - self._state_preprocessor.heading),
            torch.cos(self._target_heading - self._state_preprocessor.heading),
        )

        self._task_data[:, 0] = self._dist
        self._task_data[:, 1] = torch.cos(self._target_heading_error)
        self._task_data[:, 2] = torch.sin(self._target_heading_error)
        self._task_data[:, 3] = torch.cos(self._heading_error)
        self._task_data[:, 4] = torch.sin(self._heading_error)
        self._task_data[:, 5:7] = self._state_preprocessor.linear_velocities_body[:, :2]
        self._task_data[:, 7] = self._state_preprocessor.angular_velocities_body[:, -1]

        self._observation = torch.cat((self._task_data, actions), dim=1)
        self.check_task_completion()

    def update_goal_ROS(self, pose: PoseStamped | None = None, **kwargs) -> None:
        """Update the goal pose using the ROS message.

        When a goal is received, the task is live and the number of steps is reset.

        Args:
            position (PoseStamped): The goal pose.
        """
        if pose is not None:
            print("Received new goal")
            print(
                f"Going to position xy: {pose.pose.position.x, pose.pose.position.y}, with orientation xyzw: {pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w}"
            )

            self._target_position[0, 0] = pose.pose.position.x
            self._target_position[0, 1] = pose.pose.position.y
            self._target_heading[0, 0] = torch.atan2(
                torch.tensor(
                    2
                    * (
                        pose.pose.orientation.w * pose.pose.orientation.z
                        + pose.pose.orientation.x * pose.pose.orientation.y
                    )
                ),
                torch.tensor(1 - 2 * (pose.pose.orientation.y ** 2 + pose.pose.orientation.z ** 2)),
            )
            # A goal has been received the task is live
            self._task_is_live = True
            # Reset the number of steps to 0 when a new goal is received
            self._step = 0

    def reset(self) -> None:
        """Reset the task to its initial state."""
        super().reset()

        self._target_position.fill_(0)
        self._target_heading.fill_(0)
        self._task_data.fill_(0)
