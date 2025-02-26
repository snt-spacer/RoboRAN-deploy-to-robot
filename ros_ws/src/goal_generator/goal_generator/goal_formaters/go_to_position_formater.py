from . import Registerable
from . import BaseFormater, BaseFormaterCfg

import rclpy
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header
from dataclasses import dataclass


@dataclass
class GoToPositionFormaterCfg(BaseFormaterCfg):
    pass


class GoToPositionFormater(Registerable, BaseFormater):
    _task_cfg: GoToPositionFormaterCfg

    def __init__(
        self,
        goals_file_path: str,
        task_cfg: GoToPositionFormaterCfg = GoToPositionFormaterCfg(),
        **kwargs,
    ) -> None:
        super().__init__(goals_file_path, task_cfg, **kwargs)

        self.ROS_TYPE = PointStamped
        self.ROS_QUEUE_SIZE = 1

        self.process_yaml()

    def process_yaml(self) -> None:
        # Quick format checks
        assert "goals" in self._yaml_file, "No goals found in the YAML file."
        assert len(self._yaml_file["goals"]) > 0, "No goals found in the YAML file."
        assert "frame" in self._yaml_file, "No frame found in the YAML file."
        assert self._yaml_file["frame"].lower() in ["world", "local"], "Invalid frame coordinates type."

        self._frame = self._yaml_file["frame"].lower()
        raw_data = self._yaml_file["goals"]
        data = []
        for i in raw_data:
            goal = PointStamped()
            goal.point.x = i["position"]["x"]
            goal.point.y = i["position"]["y"]
            goal.point.z = i["position"]["z"]
            data.append(goal)
        self._goals = data
        self._iterator = self.iterator()

    def iterator(self):
        for goal in self._goals:
            yield goal

    def goal(self) -> PointStamped | None:
        self._goal = next(self._iterator, None)
        if self._goal is None:
            self._is_done = True
        return self._goal

    def log_publish(self) -> str:
        return f"Published goal: x={self._goal.point.x}, y={self._goal.point.y}, z={self._goal.point.z}"
    
    def reset(self):
        self._iterator = self.iterator()