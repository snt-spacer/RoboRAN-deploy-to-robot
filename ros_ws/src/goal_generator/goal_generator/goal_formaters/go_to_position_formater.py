from . import Registerable
from . import BaseFormater, BaseFormaterCfg

import rclpy
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header
from dataclasses import dataclass
from fp_inference.state_preprocessors import BaseStatePreProcessor

@dataclass
class GoToPositionFormaterCfg(BaseFormaterCfg):
    pass

class GoToPositionFormater(Registerable, BaseFormater):
    _task_cfg: GoToPositionFormaterCfg

    def __init__(
        self,
        goals_file_path: str,
        state_preprocessor: BaseStatePreProcessor,
        task_cfg: GoToPositionFormaterCfg = GoToPositionFormaterCfg(),
        **kwargs,
    ) -> None:
        super().__init__(goals_file_path, state_preprocessor, task_cfg)

        self.ROS_TYPE = PointStamped
        self.ROS_QUEUE_SIZE = 10

        self.process_yaml()

    def process_yaml(self):

        data = self._yaml_file["GoToPosition"]

        assert data, "No data found for GoToPosition task."
        assert data["frame"].lower() in ["world", "local"], "Invalid frame coordinates type."

        frame = data["frame"]

        self._goal = PointStamped()

        if frame.lower() == "world":
            self._goal.header = Header()
            self._goal.header.stamp = rclpy.time.Time().to_msg()
            self._goal.header.frame_id = "map"
            self._goal.point.x = data["x"]
            self._goal.point.y = data["y"]  
            self._goal.point.z = data["z"]

        self.task_completed = True

    def log_publish(self):
        return f"Published goal: x={self._goal.point.x}, y={self._goal.point.y}, z={self._goal.point.z}"