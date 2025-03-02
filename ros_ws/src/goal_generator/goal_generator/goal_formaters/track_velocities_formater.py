from . import Registerable
from . import BaseFormater

from geometry_msgs.msg import PoseArray, Pose
import numpy as np

from goal_generator.trajectory_generators import TrajectoryFactory, TrajectoryCfgFactory


class TrackVelocitiesFormater(Registerable, BaseFormater):
    """Track velocities formater.

    This class is used to define the formater for the goal generator to track velocities.
    The goal is defined as a PoseArray message. The YAML file contains the trajectory configuration.
    This configuration is used to generate the trajectory to track.
    """
    def __init__(self, goals_file_path: str, **kwargs) -> None:
        """Initialize the go through position formater.
        
        Args:
            goals_file_path (str): Path to the YAML file with the goals.
        """
        super().__init__(goals_file_path, **kwargs)

        self.ROS_TYPE = PoseArray

        self.process_yaml()

    def process_yaml(self):
        """Process the YAML file with the goals.

        The YAML file must contain the following fields:
        - trajectory: Trajectory configuration to track.

        The trajectory field must contain the fields that are required by the trajectory configuration.

        Example 1:
        trajectory:
            name: "Circle"
            cfg:
                radius: 1.0
        
        Example 2:
        trajectory:
            name: "Lissajous"
            cfg:
                A: 5.0
                B: 5.0
                a: 5.0
                b: 4.0
                omega_x: 1.570796
        """
        # Basic format checks
        assert "trajectory" in self._yaml_file, "No trajectory found in the YAML file."
        assert "name" in self._yaml_file["trajectory"], "No trajectory name found in the YAML file."
        assert "cfg" in self._yaml_file["trajectory"], "No trajectory configuration found in the YAML file."

        self._trajectory_cfg = TrajectoryCfgFactory.create(
            self._yaml_file["trajectory"]["name"], **self._yaml_file["trajectory"]["cfg"]
        )
        self._trajectory_gen = TrajectoryFactory.create(self._yaml_file["trajectory"]["name"], self._trajectory_cfg)

    def generate_pose_array(self):
        """Generate the pose array message for the trajectory."""
        positions, angles = self._trajectory_gen.trajectory

        pose_array = PoseArray()
        for xy, theta in zip(positions, angles):
            pose = Pose()
            pose.position.x = float(xy[0])
            pose.position.y = float(xy[1])
            pose.position.z = 0.0
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = float(np.sin(theta / 2))
            pose.orientation.w = float(np.cos(theta / 2))
            pose_array.poses.append(pose)
        return pose_array

    def log_publish(self) -> str:
        """Log the publish of the goal."""
        return f"Generated {self._yaml_file["trajectory"]["name"]} trajectory!"

    @property
    def goal(self) -> PoseArray:
        """Get the next goal to publish."""
        self._is_done = True
        return self.generate_pose_array()

    def reset(self):
        """Reset the formater."""
        self._is_done = False
