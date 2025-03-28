from . import Registerable
from . import BaseFormater

from geometry_msgs.msg import PoseStamped


class GoToPoseFormater(Registerable, BaseFormater):
    """Go to pose formater.
    
    This class is used to define the formater for the goal generator to go to a specific pose.
    The goal is defined as a PoseStamped message. The YAML file can contain multiple goals to go to.
    If there are multiple goals, the formater will iterate through them.
    """
    def __init__(self, goals_file_path: str, **kwargs) -> None:
        """Initialize the go to pose formater.
        
        Args:
            goals_file_path (str): Path to the YAML file with the goals.
        """
        super().__init__(goals_file_path, **kwargs)

        self.ROS_TYPE = PoseStamped

        self.process_yaml()

    def process_yaml(self) -> None:
        """Process the YAML file with the goals.

        The YAML file must contain the following fields:
        - goals: List of goals to go to.

        The goals field must contain a list of goals, where each goal is defined as:
        - position: Position of the goal (x, y, z).
        - orientation: Orientation of the goal (x, y, z, w).

        Example:
        goals:
            - position: {x: 1.0, y: 2.0, z: 0.0}
              orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
            - position: {x: 3.0, y: 4.0, z: 0.0}
              orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
            - position: {x: 5.0, y: 6.0, z: 0.0}
              orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
        """
        # Basic format checks
        assert "goals" in self._yaml_file, "No goals found in the YAML file."
        assert len(self._yaml_file["goals"]) > 0, "No GoToPose found in the YAML file."

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
        """Iterator for the goals."""
        for goal in self._goals:
            yield goal

    @property
    def goal(self) -> PoseStamped | None:
        """Get the next goal to publish.

        The method returns the next goal to publish. If there are no more goals, it returns None.

        Returns:
            PoseStamped | None: The next goal to publish.
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
        return f"Published goal: xy={self._goal.pose.position.x, self._goal.pose.position.y}, zw={self._goal.pose.orientation.z, self._goal.pose.orientation.w}"

    def reset(self):
        """Reset the formater to the initial state."""
        self._iterator = self.iterator()
        self._is_done = False
