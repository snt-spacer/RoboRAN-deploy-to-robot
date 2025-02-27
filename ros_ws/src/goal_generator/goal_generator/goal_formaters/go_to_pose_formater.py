from . import Registerable
from . import BaseFormater, BaseFormaterCfg

from geometry_msgs.msg import PoseStamped
from dataclasses import dataclass


@dataclass
class GoToPoseFormaterCfg(BaseFormaterCfg):
    pass


class GoToPoseFormater(Registerable, BaseFormater):
    _task_cfg: GoToPoseFormaterCfg

    def __init__(
        self,
        goals_file_path: str,
        task_cfg: GoToPoseFormaterCfg = GoToPoseFormaterCfg(),
        **kwargs,
    ) -> None:
        super().__init__(goals_file_path, task_cfg, **kwargs)

        self.ROS_TYPE = PoseStamped
        self.ROS_QUEUE_SIZE = 1

        self.process_yaml()

    def process_yaml(self) -> None:
        # Quick format checks
        assert "GoToPose" in self._yaml_file, "No GoToPose found in the YAML file."
        assert len(self._yaml_file["goals"]) > 0, "No GoToPose found in the YAML file."
        assert "frame" in self._yaml_file, "No frame found in the YAML file."
        assert self._yaml_file["frame"].lower() in ["world", "local"], "Invalid frame coordinates type."

        self._frame = self._yaml_file["frame"].lower()
        raw_data = self._yaml_file["goals"]
        data = []
        for i in raw_data:
            goal = PoseStamped()
            goal.pose.position.x = i["position"]["x"]
            goal.pose.position.y = i["position"]["y"]
            goal.pose.position.z = i["position"]["z"]
            goal.pose.orientation.x = i["orientation"]["x"]
            goal.pose.orientation.y = i["orientation"]["y"]
            goal.pose.orientation.z = i["orientation"]["z"]
            goal.pose.orientation.w = i["orientation"]["w"]
            data.append(goal)
        self._goals = data
        self._iterator = self.iterator()

    def iterator(self):
        for goal in self._goals:
            yield goal
    
    @property
    def goal(self) -> PoseStamped | None:
        self._goal = next(self._iterator, None)
        if self._goal is None:
            self._is_done = True
        return self._goal

    def log_publish(self) -> str:
        return f"Published goal: xy={self._goal.pose.position.x, self._goal.pose.position.y}, zw={self._goal.pose.orientation.z, self._goal.pose.orientation.w}"

    def reset(self):
        self._iterator = self.iterator()
        self._is_done = False