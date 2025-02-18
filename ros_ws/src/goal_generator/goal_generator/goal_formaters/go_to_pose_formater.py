from . import Registerable
from . import BaseFormater, BaseFormaterCfg

import rclpy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from dataclasses import dataclass

@dataclass
class GoToPoseFormaterCfg(BaseFormaterCfg):
    pass

class GoToPoseFormater(Registerable, BaseFormater):
    _task_cfg: GoToPoseFormaterCfg

    def __init__(
        self,
        goals_file_path: str = "goals.yaml",
        task_cfg: GoToPoseFormaterCfg = GoToPoseFormaterCfg(),
        **kwargs,
    ) -> None:
        super().__init__(task_cfg, goals_file_path)

        self.ROS_TYPE = PoseStamped
        self.ROS_QUEUE_SIZE = 10

        self.process_yaml()

    def process_yaml(self):

        data = self._yaml_file["GoToPose"]

        assert data, "No data found for GoToPosition task."
        assert data["frame"].lower() in ["world", "local"], "Invalid frame coordinates type."

        frame = data["frame"]

        self._goal = PoseStamped()

        if frame.lower() == "world":
            self._goal.header = Header()
            self._goal.header.stamp = rclpy.time.Time().to_msg()
            self._goal.header.frame_id = "map"
            self._goal.pose.position.x = data["x"]
            self._goal.pose.position.y = data["y"]  
            self._goal.pose.position.z = data["z"]
            self._goal.pose.orientation.x = data["qx"]
            self._goal.pose.orientation.y = data["qy"]
            self._goal.pose.orientation.z = data["qz"]
            self._goal.pose.orientation.w = data["qw"]

        self.task_completed = True

    def log_publish(self):
        return f"Published goal: xyz={self._goal.pose.position.x, self._goal.pose.position.y, self._goal.pose.position.z}, xyzw={self._goal.pose.orientation.x, self._goal.pose.orientation.y, self._goal.pose.orientation.z}"