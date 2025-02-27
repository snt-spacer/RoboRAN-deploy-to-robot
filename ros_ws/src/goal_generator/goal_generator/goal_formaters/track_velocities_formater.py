from . import Registerable
from . import BaseFormater, BaseFormaterCfg

import rclpy
from std_msgs.msg import Header
from geometry_msgs.msg import PoseArray, Pose
from dataclasses import dataclass

import torch
import numpy as np

from goal_generator.trajectory_generators import TrajectoryFactory, TrajectoryCfgFactory

@dataclass
class TrackVelocitiesFormaterCfg(BaseFormaterCfg):
    pass

class TrackVelocitiesFormater(Registerable, BaseFormater):
    _task_cfg: TrackVelocitiesFormaterCfg

    def __init__(
        self,
        goals_file_path: str,
        task_cfg: TrackVelocitiesFormaterCfg = TrackVelocitiesFormaterCfg(),
        **kwargs,
    ) -> None:
        super().__init__(goals_file_path, task_cfg)

        self.ROS_TYPE = PoseArray

        self.process_yaml()

    def process_yaml(self):
        assert "trajectory" in self._yaml_file, "No trajectory found in the YAML file."
        assert "name" in self._yaml_file["trajectory"], "No trajectory name found in the YAML file."
        assert "cfg" in self._yaml_file["trajectory"], "No trajectory configuration found in the YAML file."
        assert "frame" in self._yaml_file, "No frame found in the YAML file."
        assert self._yaml_file["frame"].lower() in ["global", "local"], "Invalid frame coordinates type."

        self._trajectory_cfg = TrajectoryCfgFactory.create(self._yaml_file["trajectory"]["name"], **self._yaml_file["trajectory"]["cfg"])
        self._trajectory_gen = TrajectoryFactory.create(self._yaml_file["trajectory"]["name"], self._trajectory_cfg)
        self._frame = self._yaml_file["frame"].lower()

    def generate_pose_array(self):
        positions, angles =  self._trajectory_gen.trajectory

        pose_array = PoseArray()
        for xy, theta in zip(positions, angles):
            pose = Pose()
            pose.position.x = float(xy[0])
            pose.position.y = float(xy[1])
            pose.position.z = 0.0
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = float(np.sin(theta/2))
            pose.orientation.w = float(np.cos(theta/2))
            pose_array.poses.append(pose)
        return pose_array
        

    def log_publish(self):
        return "Generated trajectory!"

    @property
    def goal(self) -> PoseArray:
        return self.generate_pose_array()
    
    def reset(self):
        self._is_done = False