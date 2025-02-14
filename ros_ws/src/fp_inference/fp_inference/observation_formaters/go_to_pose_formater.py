from . import Registerable
from . import BaseFormater, BaseFormaterCfg
from fp_inference.state_preprocessors import BaseStatePreProcessor

from geometry_msgs.msg import PoseStamped
from dataclasses import dataclass
from typing import Any
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
        task_cfg: GoToPoseTaskCfg = GoToPoseTaskCfg(),
        **kwargs,
    ) -> None:
        super().__init__(state_preprocessor, device, max_steps, task_cfg)

        # General parameters
        self.ROS_TYPE = PoseStamped

        self._task_data = torch.zeros((1, 6), device=self._device)
        self._target_position = torch.zeros((1, 2), device=self._device)
        self._target_heading = torch.zeros((1, 1), device=self._device)

    def build_logs(self):
        super().build_logs()
        self._logs["distance_error"] = torch.zeros((1, 1), device=self._device)
        self._logs["target_heading_error"] = torch.zeros((1, 1), device=self._device)
        self._logs["heading_error"] = torch.zeros((1, 1), device=self._device)

    def update_logs(self):
        self._logs["distance_error"] = self.dist
        self._logs["position_heading_error"] = self.target_heading_error
        self._logs["heading_error"] = self.heading_error
        self._logs["target_position"] = self._target_position
        self._logs["target_heading"] = self._target_heading

    def check_task_completion(self) -> None:
        """Check if the task has been completed."""

        if self._task_cfg.terminate_early:
            cart_dist_bool = self.dist < self._task_cfg.position_tolerance
            ang_dist_bool = torch.abs(self.target_heading_error) < self._task_cfg.heading_tolerance
        else:
            cart_dist_bool = False
            ang_dist_bool = False
        time_bool = self._step >= self._max_steps

        self._task_completed = (cart_dist_bool and ang_dist_bool) or time_bool

    def format_observation(self, actions: torch.Tensor | None = None) -> None:
        super().format_observation(actions)
        # Position distance
        self.dist = torch.linalg.norm(self._target_position - self._state_preprocessor.position[:, :2], dim=1)
        # Heading distance
        target_heading_w = torch.atan2(
            self._target_position[:, 1] - self._state_preprocessor.position[:, 1],
            self._target_position[:, 0] - self._state_preprocessor.position[:, 0],
        )
        self.target_heading_error = torch.atan2(
            torch.sin(target_heading_w - self._state_preprocessor.heading),
            torch.cos(target_heading_w - self._state_preprocessor.heading),
        )
        # Heading error (to the target heading)
        self.heading_error = torch.atan2(
            torch.sin(self._target_heading - self._state_preprocessor.heading),
            torch.cos(self._target_heading - self._state_preprocessor.heading),
        )

        self._task_data[:, 0] = self.dist
        self._task_data[:, 1] = torch.cos(self.target_heading_error)
        self._task_data[:, 2] = torch.sin(self.target_heading_error)
        self._task_data[:, 3] = torch.cos(self.heading_error)
        self._task_data[:, 4] = torch.sin(self.heading_error)
        self._task_data[:, 6:7] = self._state_preprocessor.linear_velocities_body[:, :2]
        self._task_data[:, 7] = self._state_preprocessor.angular_velocities_body[:, -1]

        self._observation = torch.cat((self._task_data, actions), dim=1)
        self.check_task_completion()

    def update_goal_ROS(self, pose: PoseStamped | None = None, **kwargs):
        """Update the goal pose using the ROS message.
        When a goal is received, the task is live and the number of steps is reset.

        Args:
            position (PoseStamped): The goal pose."""

        if pose is not None:
            self._target_position[0, 0] = pose.pose.position.x
            self._target_position[0, 1] = pose.pose.position.y
            self._target_heading[0, 0] = torch.atan2(
                2
                * (
                    pose.pose.orientation.w * pose.pose.orientation.z
                    + pose.pose.orientation.x * pose.pose.orientation.y
                ),
                1 - 2 * (pose.pose.orientation.y**2 + pose.pose.orientation.z**2),
            )
            # A goal has been received the task is live
            self._task_is_live = True
            # Reset the number of steps to 0 when a new goal is received
            self._step = 0

    def reset(self):
        super().reset()

        self._target_position.fill_(0)
        self._target_heading.fill_(0)
        self._task_data.fill_(0)
