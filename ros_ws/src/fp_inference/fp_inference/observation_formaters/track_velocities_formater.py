from . import Registerable
from . import BaseFormater, BaseFormaterCfg
from fp_inference.state_preprocessors import BaseStatePreProcessor

from custom_msgs.msg import PositionsAnglesTrackingStamped
from dataclasses import dataclass
import numpy as np
import gymnasium
import torch


@dataclass
class TrackVelocitiesFormaterCfg(BaseFormaterCfg):
    enable_linear_vel: bool = True
    enable_lateral_vel: bool = True
    enable_angular_vel: bool = True
    terminate_early: bool = False
    closed_loop: bool = True
    "Whether the trajectory is closed (it forms a loop) or not."

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

        self.ROS_TYPE = PositionsAnglesTrackingStamped
        self.current_tracking_point_indx = -1
        self.close_loop = self._task_cfg.closed_loop

        # Task parameters
        self._observation_space = gymnasium.spaces.Box(-np.inf, np.inf, (6 + num_actions,))
        self._task_data = torch.zeros((1, 6), device=self._device)
        self._target_lin_vel = torch.zeros((1, 1), device=self._device)
        self._target_lat_vel = torch.zeros((1, 1), device=self._device)
        self._target_ang_vel = torch.zeros((1, 1), device=self._device)

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
        self._logs["target_sin_cos_angles"] = torch.zeros((1, 2), device=self._device)

        self._logs_specs["target_linear_vel"] = [".m.s"]
        self._logs_specs["target_lateral_vel"] = [".m/s"]
        self._logs_specs["target_angular_vel"] = [".rad/s"]
        self._logs_specs["task_data"] = [".lin_vel_error.m/s",
                                         ".lat_vel_error.m/s",
                                         ".ang_vel_error.rad/s",
                                         ".lin_vel_body.x.m/s",
                                         ".lin_vel_body.y.m/s",
                                         ".ang_vel_body.z.rad/s"]
        self._logs_specs["target_position"] = [".x.m", ".y.m"]
        self._logs_specs["target_sin_cos_angles"] = [".sin", ".cos"]

    def update_logs(self) -> None:
        """Update the logs for the task.
        This method is called every time the logs are accessed. The logs should be updated based on the current state
        of the task. The logger performs a deep copy of the logs after this method is called, so the logs can be
        safely modified in place.
        """
        self._logs["target_linear_vel"] = self._target_lin_vel.unsqueeze(0)
        self._logs["target_lateral_vel"] = self._target_lat_vel.unsqueeze(0)
        self._logs["target_angular_vel"] = self._target_ang_vel
        self._logs["task_data"] = self._task_data
        self._logs["target_position"] = self.target_position
        self._logs["target_sin_cos_angles"] = self.target_angle

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

        self.get_velocity_vectors()

        self._target_lin_vel = torch.tensor([[1.5]], device=self._device)
        self._target_lat_vel = torch.tensor([[0.0]], device=self._device)

        # linear velocity error
        err_lin_vel = self._target_lin_vel - self._state_preprocessor.linear_velocities_body[:, 0]

        # lateral velocity error
        err_lat_vel = self._target_lat_vel - self._state_preprocessor.linear_velocities_body[:, 1]

        # angular velocity error
        err_ang_vel = self._target_ang_vel - self._state_preprocessor.angular_velocities_body[:, -1]

        self._task_data[:, 0] = err_lin_vel * self._task_cfg.enable_linear_vel
        self._task_data[:, 1] = err_lat_vel * self._task_cfg.enable_lateral_vel
        self._task_data[:, 2] = err_ang_vel * self._task_cfg.enable_angular_vel
        self._task_data[:, 3:5] = self._state_preprocessor.linear_velocities_body[:, :2]
        self._task_data[:, 5] = self._state_preprocessor.angular_velocities_body[:, -1]

        print("Task data: ", self._task_data)

        self._observation = torch.cat((self._task_data, actions), dim=1)
        self.check_task_completion()

    def update_goal_ROS(self, ros_data: PositionsAnglesTrackingStamped | None = None, **kwargs) -> None:
        if ros_data is not None:
            print("Received new target velocities")
            self._step += 1
            self.positions = torch.tensor([ros_data.positions_x, ros_data.positions_y], device=self._device).T
            self.angles = torch.tensor([ros_data.angles_sin, ros_data.angles_cos], device=self._device).T
            self.target_tracking_velocity = ros_data.target_tracking_velocity
            # A goal has been received the task is live
            self._task_is_live = True
            # Reset the number of steps to 0 when a new goal is received
            self._step = 0

    def reset(self) -> None:
        """Reset the task to its initial state."""
        super().reset()
        self._target_lin_vel.fill_(0)
        self._target_lat_vel.fill_(0)
        self._target_ang_vel.fill_(0)
        self._task_data.fill_(0)


    def get_velocity_vectors(self):
        self.get_tracking_point_indx()
        self.target_position, self.target_angle = self.get_point_for_tracking()
        print("Target position: ", self.target_position)
        direction_vector = self.compute_direction_vector()
        velocity_goal_global = direction_vector * self.target_tracking_velocity
        velocity_goal = (torch.linalg.pinv(self._state_preprocessor.rotation_matrix[0]) @ velocity_goal_global[0]).unsqueeze(0)
        print("Velocity goal: ", velocity_goal)
        self._target_lin_vel = velocity_goal[:, 0]
        self._target_lat_vel = velocity_goal[:, 1]

    def get_tracking_point_indx(self):
        distances = torch.linalg.norm(self.positions - self._state_preprocessor.position[:, :2], dim=-1)
        if self.current_tracking_point_indx == -1:
            self.current_tracking_point_indx = 0
        else:
            indices = torch.where(distances < 0.1)[0]
            if len(indices) > 0:
                indices = indices[indices < 60]
                if len(indices) > 0:
                    self.current_tracking_point_indx = torch.max(indices).item()

    def get_point_for_tracking(self):
        position = self.positions[self.current_tracking_point_indx]
        angle = self.angles[self.current_tracking_point_indx]
        self.roll_trajectory()
        return position, angle

    def roll_trajectory(self):
        if self.close_loop:
            self.positions = torch.roll(self.positions, -self.current_tracking_point_indx, dims=0)
            self.angles = torch.roll(self.angles, -self.current_tracking_point_indx, dims=0)
            self.current_tracking_point_indx = 0
        else:
            self.positions = self.positions[self.current_tracking_point_indx:]
            self.angles = self.angles[self.current_tracking_point_indx:]
            self.current_tracking_point_indx = 0

        if self.positions.shape[0] == 0:
            self.task_completed = True

    def compute_direction_vector(self):
        diff = self.target_position - self._state_preprocessor.position[:, :2]
        #diff = (self._state_preprocessor.rotation_matrix[0].T @ (self.target_position - self._state_preprocessor.position[:, :2]).T).T
        #print("target_position: ", self.target_position)
        #print("current_position: ", self._state_preprocessor.position[:, :2])
        #print("current_position_local", diff)
        return diff / torch.linalg.norm(diff)
