from . import Registerable
from . import BaseFormater, BaseFormaterCfg

import rclpy
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Header
from dataclasses import dataclass

import torch
from fp_inference.state_preprocessors import BaseStatePreProcessor
import numpy as np


@dataclass
class TrackVelocitiesFormaterCfg(BaseFormaterCfg):
    closed_loop: bool = True
    "Whether the trajectory is closed (it forms a loop) or not."
    trajectory_x_offset: float = 0.0
    "Offset in x direction for the trajectory."
    trajectory_y_offset: float = 0.0
    "Offset in y direction for the trajectory."
    target_tracking_velocity: float = 0.25


class TrackVelocitiesFormater(Registerable, BaseFormater):
    _task_cfg: TrackVelocitiesFormaterCfg

    def __init__(
        self,
        goals_file_path: str,
        state_preprocessor: BaseStatePreProcessor,
        task_cfg: TrackVelocitiesFormaterCfg = TrackVelocitiesFormaterCfg(),
        **kwargs,
    ) -> None:
        super().__init__(goals_file_path, state_preprocessor, task_cfg)

        self.ROS_TYPE = TwistStamped
        self.ROS_QUEUE_SIZE = 10
        self.current_tracking_point_indx = -1
        self.close_loop = self._task_cfg.closed_loop

        self.process_yaml()

    def process_yaml(self):

        data = self._yaml_file["TrackVelocities"]

        assert data, "No data found for TrackVelocities task."
        assert data["frame"].lower() in ["world", "local"], "Invalid frame coordinates type."

        frame = data["frame"]
        self.close_loop = data["closed"]

        if data["shape"].lower() == "circle":
            self.generate_circle(data["circle"])
        elif data["shape"].lower() == "square":
            self.generate_square(data["square"])
        elif data["shape"].lower() == "spiral":
            self.generate_spiral(data["spiral"])
        else:
            raise ValueError("Invalid shape of velocities.")

        # TODO check for frame world or local coords
        print("Hereeeeeeeee")
        self.get_velocity_vector()

    def generate_circle(self, data):
        radius = data["radius"]
        num_points = data["num_points"]
        offset = torch.tensor([self._task_cfg.trajectory_x_offset, self._task_cfg.trajectory_y_offset])

        theta = torch.linspace(0, 2 * np.pi, num_points, endpoint=(not self._task_cfg.closed_loop))
        self.positions = torch.stack([torch.cos(theta) * radius, torch.sin(theta) * radius], dim=-1) + offset
        self.angles = torch.stack([-torch.sin(theta), torch.cos(theta)], dim=-1)

    def generate_square(self, data):
        pass

    def generate_spiral(self, data):
        pass

    def get_velocity_vector(self):
        self.get_tracking_point_indx()
        self.target_position, self.target_angle = self.get_point_for_tracking()
        direction_vector = self.compute_direction_vector()
        velocity_goal = direction_vector * self._task_cfg.target_tracking_velocity
        print(f"Velocity goal: {velocity_goal}")

    def get_tracking_point_indx(self):
        distances = torch.linalg.norm(self.positions - self._state_preprocessor.robot_position[:2], dim=-1)
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
            self.positions = torch.roll(self.positions, -self.current_tracking_point_indx, axis=0)
            self.angles = torch.roll(self.angles, -self.current_tracking_point_indx, axis=0)
            self.current_tracking_point_indx = 0
        else:
            self.positions = self.positions[self.current_tracking_point_indx :]
            self.angles = self.angles[self.current_tracking_point_indx :]
            self.current_tracking_point_indx = 0

        if self.positions.shape[0] == 0:
            self.task_completed = True

    def compute_direction_vector(self):
        diff = self.target_position - self._state_preprocessor.robot_position[:2]
        return diff / torch.linalg.norm(diff)

    def log_publish(self):
        return f"Published goal: x={self._goal.twist.linear.x}, y={self._goal.twist.linear.y}, z={self._goal.twist.linear.z}"
