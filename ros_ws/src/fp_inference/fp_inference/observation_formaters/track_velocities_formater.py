from . import Registerable
from . import BaseFormater, BaseFormaterCfg
from fp_inference.state_preprocessors import BaseStatePreProcessor

from geometry_msgs.msg import TwistStamped
from dataclasses import dataclass
import numpy as np
import gymnasium
import torch

@dataclass
class TrackVelocitiesTaskCfg(BaseFormaterCfg):
    enable_linear_vel: bool = True
    enable_lateral_vel: bool = True
    enable_angular_vel: bool = True
    terminate_early: bool = False

class TrackVelocitiesTask(Registerable, BaseFormater):
    _task_cfg: TrackVelocitiesTaskCfg

    def __init__(
        self,
        state_preprocessor: BaseStatePreProcessor | None = None,
        device: str = "cuda",
        max_steps: int = 500,
        num_actions: int = 2,
        task_cfg: TrackVelocitiesTaskCfg = TrackVelocitiesTaskCfg(),
        **kwargs,
    ) -> None:
        super().__init__(state_preprocessor, device, max_steps, task_cfg)

        self.ROS_TYPE = TwistStamped

        # Task parameters
        self._observation_space = gymnasium.spaces.Box(-np.inf, np.inf, (6 + num_actions,))

        self._task_data = torch.zeros((1, 6), device=self._device)
        self._target_lin_vel = torch.zeros((1, 1), device=self._device)
        self._target_lat_vel = torch.zeros((1, 1), device=self._device)
        self._target_ang_vel = torch.zeros((1, 1), device=self._device)

    def build_logs(self):
        super().build_logs()
        self._logs["target_linear_vel"] = torch.zeros((1,1), device=self._device)
        self._logs["target_lateral_vel"] = torch.zeros((1,1), device=self._device)
        self._logs["target_angular_vel"] = torch.zeros((1,1), device=self._device)
        self._logs["task_data"] = torch.zeros((1, 8), device=self._device)
        self._logs_specs["linear_vel_error"] = [".m.s"]
        self._logs_specs["lateral_vel_error"] = [".m/s"]
        self._logs_specs["angular_vel_error"] = [".rad/s"]
        self._logs_specs["task_data"] = [".lin_vel_error.m/s",
                                         ".lat_vel_error.m/s",
                                         ".ang_vel_error.rad/s",
                                         ".lin_vel_body.x.m/s",
                                         ".lin_vel_body.y.m/s",
                                         ".ang_vel_body.z.rad/s"]


    def update_logs(self):
        self._logs["target_lin_vel"] = self._target_lin_vel
        self._logs["target_lat_vel"] = self._target_lat_vel
        self._logs["target_ang_vel"] = self._target_ang_vel
        self._logs["task_data"] = self._task_data

    def check_task_completition(self) -> None:
        """ Check if the task has been completed."""
        time_bool = self._step >= self._max_steps
        self._task_completed = time_bool

    def format_observation(self, actions: torch.Tensor | None = None) -> None:
        super().format_observation(actions)

        # linear velocity error
        err_lin_vel = self._target_lin_vel - self._state_preprocessor.linear_velocities_body[:, 0]

        # lateral velocity error
        err_lat_vel = self._target_lat_vel - self._state_preprocessor.linear_velocities_body[:, 1]

        # angular velocity error
        err_ang_vel = self._target_ang_vel - self._state_preprocessor.angular_velocities_body

        self._task_data[:, 0] = err_lin_vel * self._task_cfg.enable_linear_vel
        self._task_data[:, 1] = err_lat_vel * self._task_cfg.enable_lateral_vel
        self._task_data[:, 2] = err_ang_vel * self._task_cfg.enable_angular_vel
        self._task_data[:, 3,5] = self._state_preprocessor.linear_velocities_body[:, :2]
        self._task_data[:, 5] = self._state_preprocessor.angular_velocities_body[:, -1]

        self._observation = torch.cat((self._task_data, actions), dim=1)
        self.check_task_completion()

    def update_goal_ROS(self, velocity: TwistStamped | None = None, **kwargs) -> None:
        if velocity is not None:
            print("Received new target velocities")
            print(f"Linear velocity: {velocity.twist.linear.x}, {velocity.twist.linear.y}, Angular velocity: {velocity.twist.angular.z}")
            self._step += 1
            self._target_lin_vel = torch.tensor([velocity.twist.linear.x], device=self._device)
            self._target_lat_vel = torch.tensor([velocity.twist.linear.y], device=self._device)
            self._target_ang_vel = torch.tensor([velocity.twist.angular.z], device=self._device)
            # A goal has been received the task is live
            self._task_is_live = True
            # Reset the number of steps to 0 when a new goal is received
            self._step = 0
            
    def reset(self):
        super().reset()
        self._target_lin_vel.fill_(0)
        self._target_lat_vel.fill_(0)
        self._target_ang_vel.fill_(0)
        self._task_data.fill_(0)
