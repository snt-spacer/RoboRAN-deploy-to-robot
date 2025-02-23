from . import Registerable
from . import BaseFormater, BaseFormaterCfg

import rclpy
from std_msgs.msg import Header
from custom_msgs.msg import PositionsAnglesTrackingStamped
from dataclasses import dataclass

import torch
import numpy as np


@dataclass
class TrackVelocitiesFormaterCfg(BaseFormaterCfg):
    closed_loop: bool = True
    "Whether the trajectory is closed (it forms a loop) or not."
    trajectory_x_offset: float = 0.0
    "Offset in x direction for the trajectory."
    trajectory_y_offset: float = 0.0
    "Offset in y direction for the trajectory."


class TrackVelocitiesFormater(Registerable, BaseFormater):
    _task_cfg: TrackVelocitiesFormaterCfg

    def __init__(
        self,
        goals_file_path: str,
        task_cfg: TrackVelocitiesFormaterCfg = TrackVelocitiesFormaterCfg(),
        **kwargs,
    ) -> None:
        super().__init__(goals_file_path, task_cfg)

        self.ROS_TYPE = PositionsAnglesTrackingStamped
        self.ROS_QUEUE_SIZE = 1
        self.current_tracking_point_indx = -1
        self.close_loop = self._task_cfg.closed_loop

        self.process_yaml()

    def process_yaml(self):

        data = self._yaml_file["TrackVelocities"]

        assert data, "No data found for TrackVelocities task."
        assert data["frame"].lower() in ["world", "local"], "Invalid frame coordinates type."

        frame = data["frame"]
        self.close_loop = data["closed"]

        self._goal = PositionsAnglesTrackingStamped()

        if data["shape"].lower() == "circle":
            self.generate_circle(data["circle"])
            self.shape_type = "circle"
        elif data["shape"].lower() == "square":
            self.generate_square(data["square"])
            self.shape_type = "square"
        elif data["shape"].lower() == "spiral":
            self.generate_spiral(data["spiral"])
            self.shape_type = "spiral"
        else:
            raise ValueError("Invalid shape of velocities.")
        

        if frame.lower() == "world":
            self._goal.header = Header()
            self._goal.header.stamp = rclpy.time.Time().to_msg()
            self._goal.header.frame_id = "map"

            self._goal.positions_x = self.positions[:, 0].flatten().tolist() 
            self._goal.positions_y = self.positions[:, 1].flatten().tolist() 
            self._goal.angles_sin = self.angles[:, 0].flatten().tolist()
            self._goal.angles_cos = self.angles[:, 1].flatten().tolist()
            self._goal.target_tracking_velocity = data["target_tracking_velocity"]

        self.task_completed = True

    def log_publish(self):
        return f"Published goal: {self.shape_type}"

    def generate_circle(self, data):
        radius = data["radius"]
        num_points = data["num_points"]
        offset = torch.tensor([self._task_cfg.trajectory_x_offset, self._task_cfg.trajectory_y_offset])

        theta = torch.linspace(0, 2 * 3.1415926535897932384, steps=num_points)
        self.positions = torch.stack([torch.cos(theta) * radius, torch.sin(theta) * radius], dim=-1) + offset
        self.angles = torch.stack([-torch.sin(theta), torch.cos(theta)], dim=-1)

    def generate_square(self, data):
        pass

    def generate_spiral(self, data):
        pass
