from . import Registerable
from . import BaseFormater

from geometry_msgs.msg import PoseArray, Pose


class GoThroughPositionsFormater(Registerable, BaseFormater):
    """Go through position formater.
    
    This class is used to define the formater for the goal generator to go through a set of positions.
    The goal is defined as a PoseArray message. The YAML file can contain multiple sets of positions to go through.
    If there are multiple sets of positions, the formater will iterate through them.
    """
    def __init__(self, goals_file_path: str, **kwargs) -> None:
        """Initialize the go through position formater.
        
        Args:
            goals_file_path (str): Path to the YAML file with the goals.
        """
        super().__init__(goals_file_path, **kwargs)

        self.ROS_TYPE = PoseArray

        self.process_yaml()

    def process_yaml(self) -> None:
        """Process the YAML file with the goals.
        
        The YAML file must contain the following fields:
        - goals: List of sets of positions to go through.
        
        The goals field must contain a list of sets of positions, where each set is defined as:
        - positions: List of positions to go through.
        
        In each set of positions, each position is defined as:
        - position: Position of the goal (x, y, z).

        Example:
        goals:
            - positions:
                - position: {x: 1.0, y: 2.0, z: 0.0}
                - position: {x: 3.0, y: 4.0, z: 0.0}
                - position: {x: 5.0, y: 6.0, z: 0.0}
            - positions:
                - position: {x: 7.0, y: 8.0, z: 0.0}
                - position: {x: 9.0, y: 10.0, z: 0.0}
                - position: {x: 11.0, y: 12.0, z: 0.0}
        """
        # Basic format checks
        assert "goals" in self._yaml_file, "No goals found in the YAML file."
        assert len(self._yaml_file["goals"]) > 0, "No goals found in the YAML file."

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
        """Iterator for the goals."""
        for goal in self._goals:
            yield goal

    @property
    def goal(self) -> PoseArray | None:
        """Get the next goal to publish.

        The method returns the next goal to publish. If there are no more goals to publish, it returns None.

        Returns:
            PoseArray | None: The next goal to publish.
        """
        self._goal = next(self._iterator, None)
        if self._goal is None:
            self._is_done = True
        return self._goal

    def log_publish(self) -> str:
        """Log the published goal.

        Returns:
            str: The published goal as a string.
        """
        return f"Published goal:\n + x={[pose.position.x for pose in self._goal.poses]}, y={[pose.position.y for pose in self._goal.poses]}"

    def reset(self):
        """Reset the formater to the initial state."""
        self._iterator = self.iterator()
        self._is_done = False
