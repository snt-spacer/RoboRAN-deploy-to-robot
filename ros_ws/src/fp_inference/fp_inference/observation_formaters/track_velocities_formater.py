from . import Registerable
from . import Registerable
from . import BaseFormater, BaseFormaterCfg
from fp_inference.state_preprocessors import BaseStatePreProcessor

from geometry_msgs.msg import PoseArray
from dataclasses import dataclass
import numpy as np
import gymnasium
import torch


@dataclass
class TrackVelocitiesFormaterCfg(BaseFormaterCfg):
    enable_linear_vel: bool = True
    enable_lateral_vel: bool = False
    enable_angular_vel: bool = True
    terminate_early: bool = False
    closed_loop: bool = True
    "Whether the trajectory is closed (it forms a loop) or not."

    max_lin_vel: float = 0.3
    max_ang_vel: float = 0.5

class TrackVelocitiesFormater(Registerable, BaseFormater):
    _task_cfg: TrackVelocitiesFormaterCfg

    def __init__(
        self,
        state_preprocessor: BaseStatePreProcessor | None = None,
        device: str = "cuda",
        max_steps: int = 500,
        num_actions: int = 2,
        task_cfg: TrackVelocitiesFormaterCfg = TrackVelocitiesFormaterCfg(),
        **kwargs,
    ) -> None:
        """Task formatter for the track velocities task."""
        super().__init__(state_preprocessor, device, max_steps, task_cfg)

        self.ROS_TYPE = PoseArray
        self.current_tracking_point_indx = -1
        self.close_loop = self._task_cfg.closed_loop

        # Task parameters
        self._observation_space = gymnasium.spaces.Box(-np.inf, np.inf, (6 + num_actions,))
        self._task_data = torch.zeros((1, 6), device=self._device)
        self._target_lin_vel_b = torch.zeros((1, 1), device=self._device)
        self._target_lat_vel_b = torch.zeros((1, 1), device=self._device)
        self._target_ang_vel = torch.zeros((1, 1), device=self._device)
        self._target_position = torch.zeros((1, 2), device=self._device)
        self._target_heading = torch.zeros((1, 1), device=self._device)

        self.current_point = -1
        self.lookhead = 0.5
        self.closed = self._task_cfg.closed_loop

    def build_logs(self) -> None:
        """Build the logs for the task.

        Logs are used by the logger to automatically keep track of the task's progress. This provides a unified way
        to keep track of the task's progress and to compare different tasks and robots.
        """
        super().build_logs()

        # Task logs
        self._logs["target_linear_vel"] = torch.zeros((1, 1), device=self._device)
        self._logs["target_lateral_vel"] = torch.zeros((1, 1), device=self._device)
        self._logs["target_angular_vel"] = torch.zeros((1, 1), device=self._device)
        self._logs["task_data"] = torch.zeros((1, 8), device=self._device)
        self._logs["target_position"] = torch.zeros((1, 2), device=self._device)
        self._logs["target_heading"] = torch.zeros((1, 1), device=self._device)

        self._logs_specs["target_linear_vel"] = [".m/s"]
        self._logs_specs["target_lateral_vel"] = [".m/s"]
        self._logs_specs["target_angular_vel"] = [".rad/s"]
        self._logs_specs["task_data"] = [".lin_vel_error.m/s",
                                         ".lat_vel_error.m/s",
                                         ".ang_vel_error.rad/s",
                                         ".lin_vel_body.x.m/s",
                                         ".lin_vel_body.y.m/s",
                                         ".ang_vel_body.z.rad/s"]
        self._logs_specs["target_position"] = [".x.m", ".y.m"]
        self._logs_specs["target_heading"] = [".rad"]

    def update_logs(self) -> None:
        """Update the logs for the task.
        This method is called every time the logs are accessed. The logs should be updated based on the current state
        of the task. The logger performs a deep copy of the logs after this method is called, so the logs can be
        safely modified in place.
        """
        self._logs["target_linear_vel"] = self._target_lin_vel_b
        self._logs["target_lateral_vel"] = self._target_lat_vel_b
        self._logs["target_angular_vel"] = self._target_ang_vel
        self._logs["task_data"] = self._task_data
        self._logs["target_position"] = self._target_position

    def check_task_completion(self) -> None:
        """ Check if the task has been completed."""
        time_bool = self._step >= self._max_steps
        self._task_completed = time_bool

    def format_observation(self, actions: torch.Tensor | None = None) -> None:
        """Format the observation for the task.

        Args:
            actions (torch.Tensor): The actions to be formatted with the observation.
        """
        super().format_observation(actions)

        # Get velocities vectors
        distances = torch.linalg.norm(self.target_positions - self._state_preprocessor.position[:, :2], dim=1)
        if self.current_point == -1:
            self.current_point = 0
        else:
            indices = torch.where(distances < self.lookhead)[0]
            if len(indices) > 0:
                indices = indices[indices < 60]
                if len(indices) > 0:
                    self.current_point = torch.max(indices).item()

        self._target_position = self.target_positions[self.current_point].unsqueeze(0)
        
        target_lin_vel_b = self._state_preprocessor.rotation_matrix[0] @ (self._target_position[0] - self._state_preprocessor.position[0, :2]).T
        
        norm = torch.linalg.norm(target_lin_vel_b)
        if norm < 1e-3:
            target_lin_vel_b.fill_(0)
        else:
            target_lin_vel_b = target_lin_vel_b / norm

        target_ang_vel_b = self._target_heading = torch.arctan2(target_lin_vel_b[1], target_lin_vel_b[0])
        
        self._target_lin_vel_b[:,0] = target_lin_vel_b[0] * self._task_cfg.max_lin_vel
        self._target_lat_vel_b[:,0] = 0.0
        self._target_ang_vel[:,0] = torch.clamp(target_ang_vel_b * self._task_cfg.max_ang_vel, -0.5, 0.5)

        self.roll_trajectory()

        # linear velocity error
        err_lin_vel = self._target_lin_vel_b - self._state_preprocessor.linear_velocities_body[:, 0]

        # lateral velocity error
        err_lat_vel = self._target_lat_vel_b - self._state_preprocessor.linear_velocities_body[:, 1]

        # angular velocity error
        err_ang_vel = self._target_ang_vel - self._state_preprocessor.angular_velocities_body[:, -1]

        self._task_data[:, 0] = err_lin_vel * self._task_cfg.enable_linear_vel
        self._task_data[:, 1] = err_lat_vel * self._task_cfg.enable_lateral_vel
        self._task_data[:, 2] = err_ang_vel * self._task_cfg.enable_angular_vel
        self._task_data[:, 3:5] = self._state_preprocessor.linear_velocities_body[:, :2]
        self._task_data[:, 5] = self._state_preprocessor.angular_velocities_body[:, -1]

        self._observation = torch.cat((self._task_data, actions), dim=1)
        self.check_task_completion()

    def update_goal_ROS(self, ros_data: PoseArray | None = None, **kwargs) -> None:
        if ros_data is not None:
            print("Received new target velocities")
            self._step += 1
            data = [[pose.position.x, pose.position.y] for pose in ros_data.poses]
            self.target_positions = torch.tensor(data, device=self._device)
            # A goal has been received the task is live
            self._task_is_live = True
            # Reset the number of steps to 0 when a new goal is received
            self._step = 0

    def reset(self) -> None:
        """Reset the task to its initial state."""
        super().reset()
        self._target_lin_vel_b.fill_(0)
        self._target_lat_vel_b.fill_(0)
        self._target_ang_vel.fill_(0)
        self._task_data.fill_(0)


    def roll_trajectory(self) -> None:
        if self.closed:
            self.target_positions = torch.roll(self.target_positions, shifts=-self.current_point, dims=0)
            self.current_point = 0
        else:
            self.target_positions = self.target_positions[self.current_point:]
            self.current_point = 0 
        if self.target_positions.shape[0] <= 1:
            self.is_done = True