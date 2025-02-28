from . import Registerable
from . import BaseFormater, BaseFormaterCfg

from geometry_msgs.msg import PoseArray, Pose
from dataclasses import dataclass


@dataclass
class GoThroughPositionFormaterCfg(BaseFormaterCfg):
    pass


class GoThroughPositionFormater(Registerable, BaseFormater):
    _task_cfg: GoThroughPositionFormaterCfg

    def __init__(
        self,
        goals_file_path: str,
        task_cfg: GoThroughPositionFormaterCfg = GoThroughPositionFormaterCfg(),
        **kwargs,
    ) -> None:
        super().__init__(goals_file_path, task_cfg, **kwargs)

        self.ROS_TYPE = PoseArray

        self.process_yaml()

    def process_yaml(self) -> None:
        # Quick format checks
        assert "goals" in self._yaml_file, "No goals found in the YAML file."
        assert len(self._yaml_file["goals"]) > 0, "No goals found in the YAML file."
        assert "frame" in self._yaml_file, "No frame found in the YAML file."
        assert self._yaml_file["frame"].lower() in ["global", "local"], "Invalid frame coordinates type."

        self._frame = self._yaml_file["frame"].lower()
        raw_data = self._yaml_file["goals"]
        data = []
        for i in raw_data:
            assert len(i["positions"]) > 0, "No points given in the set of points to track."
            goal = PoseArray()
            for j in i["positions"]:
                goal_point = Pose()
                goal_point.position.x = j["x"]
                goal_point.position.y = j["y"]
                goal_point.position.z = j["z"]
                goal.poses.append(goal_point)
            data.append(goal)
        self._goals = data
        self._iterator = self.iterator()

    def iterator(self):
        for goal in self._goals:
            yield goal

    @property
    def goal(self) -> PoseArray | None:
        self._goal = next(self._iterator, None)
        if self._goal is None:
            self._is_done = True
        return self._goal

    def log_publish(self) -> str:
        return f"Published goal:\n + x={[pose.position.x for pose in self._goal.poses]}, y={[pose.position.y for pose in self._goal.poses]}"

    def reset(self):
        self._iterator = self.iterator()
        self._is_done = False
