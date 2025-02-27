from . import Registerable
from . import BaseFormater, BaseFormaterCfg
from fp_inference.state_preprocessors import BaseStatePreProcessor

from geometry_msgs.msg import PoseArray
from dataclasses import dataclass
import numpy as np
import gymnasium
import torch


# Need to test if it loops.
# Need to test if it allows for early termination.
# Need to test with a single goal.

@dataclass
class GoThroughPositionsTaskCfg(BaseFormaterCfg):
    """Configuration for the go to position task."""

    position_tolerance: float = 0.3
    loop_through_goals: bool = False
    num_goals_in_obs: int = 2


class GoThroughPositionsFormater(Registerable, BaseFormater):
    """Task formatter for the go to position task."""

    _task_cfg: GoThroughPositionsTaskCfg

    def __init__(
        self,
        state_preprocessor: BaseStatePreProcessor | None = None,
        device: str = "cuda",
        max_steps: int = 500,
        task_cfg: GoThroughPositionsTaskCfg = GoThroughPositionsTaskCfg(),
        num_actions: int = 2,
        **kwargs,
    ) -> None:
        """Task formatter for the go to position task."""
        super().__init__(state_preprocessor, device, max_steps, task_cfg)

        # General parameters
        self.ROS_TYPE = PoseArray

        # Task parameters
        self._observation_space = gymnasium.spaces.Box(-np.inf, np.inf, (6 + 3 * (self._task_cfg.num_goals_in_obs - 1)+ num_actions,))
        self._task_data = torch.zeros((1, 6 + 3 * (self._task_cfg.num_goals_in_obs - 1)), device=self._device)
        self._target_position = torch.zeros((1, 2), device=self._device)
        self._target_positions = None
        self._num_goals_reached = torch.zeros((1, 1), device=self._device)
        self._current_goal_idx = 0

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
        self._logs["num_goals_reached"] = torch.tensor((1, 1), device=self._device)
        self._logs["task_data"] = torch.zeros((1, 6), device=self._device)

        # Log specifications
        self._logs_specs["distance_error"] = [".m"]
        self._logs_specs["target_heading_error"] = [".rad"]
        self._logs_specs["target_position"] = [".x.m", ".y.m"]
        self._logs_specs["num_goals_reached"] = [".u"]
        self._logs_specs["task_data"] = [
            ".dist.m",
            ".cos(heading).u",
            ".sin(heading).u",
            ".lin_vel_body.x.m/s",
            ".lin_vel_body.y.m/s",
            ".ang_vel_body.rad/s",
        ]
        for i in range(self._task_cfg.num_goals_in_obs - 1):
            self._logs_specs["task_data"] += [
                ".goal_"+str(i+1)+"_dist.m",
                ".cos(goal_"+str(i+1)+"_heading).u",
                ".sin(goal_"+str(i+1)+"_heading).u",
            ]

    def update_logs(self) -> None:
        """Update the logs for the task.

        This method is called every time the logs are accessed. The logs should be updated based on the current state
        of the task. The logger performs a deep copy of the logs after this method is called, so the logs can be
        safely modified in place.
        """
        self._logs["distance_error"] = self._position_dist
        self._logs["target_heading_error"] = self._target_heading_error
        self._logs["target_position"] = self._target_position
        self._logs["num_goals_reached"] = self._num_goals_reached
        self._logs["task_data"] = self._task_data

    def check_task_completion(self) -> None:
        """Check if the task has been completed."""
        cart_dist_bool = self._position_dist < self._task_cfg.position_tolerance
        if cart_dist_bool:
            self._num_goals_reached[0, 0] += 1
            self._current_goal_idx  = (self._current_goal_idx + 1) % self._target_positions.shape[1]
            self._target_position[0, :] = self._target_positions[0, self._current_goal_idx, :]

        terminate = False
        if not self._task_cfg.loop_through_goals:
            if self._num_goals_reached[0, 0] == self._target_positions.shape[1]:
                terminate = True

        time_bool = self._step >= self._max_steps

        self._task_completed = terminate or time_bool

    def format_observation(self, actions: torch.Tensor | None = None) -> None:
        """Format the observation for the task.

        Args:
            actions (torch.Tensor): The actions to be formatted with the observation.
        """
        super().format_observation(actions)

        # position error
        self._position_dist = torch.linalg.norm(
            self._target_position - self._state_preprocessor.position[:, :2], dim=1, keepdim=True
        )
        heading = self._state_preprocessor.heading
        # Heading distance
        target_heading_w = torch.atan2(
            self._target_position[:, 1] - self._state_preprocessor.position[:, 1],
            self._target_position[:, 0] - self._state_preprocessor.position[:, 0],
        )
        self._target_heading_error = torch.atan2(
            torch.sin(target_heading_w - self._state_preprocessor.heading),
            torch.cos(target_heading_w - self._state_preprocessor.heading),
        )

        # Store in buffer
        self._task_data[:, 0:2] = self._state_preprocessor.linear_velocities_body[:, :2]
        self._task_data[:, 2] = self._state_preprocessor.angular_velocities_body[:, -1]
        self._task_data[:, 3] = self._position_dist
        self._task_data[:, 4] = torch.cos(self._target_heading_error)
        self._task_data[:, 5] = torch.sin(self._target_heading_error)
        # We compute the observations of the subsequent goals in the robot frame as the goals are not oriented.
        for i in range(self._task_cfg.num_goals_in_obs - 1):
            # Check if the index is looking beyond the number of goals
            overflowing = (self._current_goal_idx + i + 1) >= len(self._target_positions)
            # If it is, then set the next index to 0 (Loop around)
            index = (self._current_goal_idx + i + 1) * (not overflowing)
            # Compute the distance between the nth goal, and the robot
            goal_distance = torch.linalg.norm(
                self._state_preprocessor.position[:, :2] - self._target_positions[:, index],
                dim=-1,
                keepdim=True,
            )
            # Compute the heading distance between the nth goal, and the robot
            target_heading_w = torch.atan2(
                self._target_positions[:, index, 1] - self._state_preprocessor.position[:, 1],
                self._target_positions[:, index, 0] - self._state_preprocessor.position[:, 0],
            )
            target_heading_error = torch.atan2(
                torch.sin(target_heading_w - heading),
                torch.cos(target_heading_w - heading),
            )
            # If the task is not set to loop, we set the next goal to be 0.
            if not self._task_cfg.loop_through_goals:
                goal_distance = goal_distance * (not overflowing)
                target_heading_error = target_heading_error * (not overflowing)
            # Add to buffer
            self._task_data[:, 6 + 3 * i] = goal_distance
            self._task_data[:, 7 + 3 * i] = torch.cos(target_heading_error)
            self._task_data[:, 8 + 3 * i] = torch.sin(target_heading_error)

        self._observation = torch.cat((self._task_data, actions), dim=1)
        self.check_task_completion()


    def update_goal_ROS(self, positions: PoseArray | None = None, **kwargs) -> None:
        """Update the goal position using the ROS message.

        When a goal is received, the task is live and the number of steps is reset.

        Args:
            position (PointStamped): The goal position.
        """
        self._target_positions = torch.zeros((1, len(positions.poses), 2), device=self._device)
        if positions is not None:
            print("Received new goals")
            print("Going to positions: ")
            for i, position in enumerate(positions.poses):
                print(" - Goal #"+str(i)+":",position.position.x, position.position.y)
                self._target_positions[:, i, 0] = position.position.x
                self._target_positions[:, i, 1] = position.position.y
            self._num_goals_reached.fill_(0)
            self._current_goal_idx = 0

            self._step += 1
            self._target_position[0, :] = self._target_positions[0, 0, :]
            # A goal has been received the task is live
            self._task_is_live = True
            # Reset the number of steps to 0 when a new goal is received
            self._step = 0

    def reset(self) -> None:
        """Reset the task to its initial state."""
        super().reset()

        self._target_position.fill_(0)
        self._task_data.fill_(0)
        self._num_goals_reached.fill_(0)
        self._current_goal = 0
