from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from typing import Any
import yaml

class BaseFormater:
    """Base goal formater class.
    
    This class is used to define the base formater class for the goal generator.
    All goal formater classes must inherit from this class, and implement the following methods:
    - process_yaml: Process the YAML file with the goals.
    - log_publish: Log the published goal.
    - reset: Reset the formater to the initial state.
    """
    def __init__(self, goals_file_path: str) -> None:
        """Initialize the base formater class.

        This method initializes the base formater class with the path to the YAML file with the goals.
        It also initializes the ROS type and QoS profile for the goal publisher. The child classes must
        override the ROS_TYPE with the correct ROS message type to be published.
        
        Args:
            goals_file_path (str): Path to the YAML file with the goals.
        """
        self._goals_file_path = goals_file_path
        self._goal = None

        self.ROS_TYPE = None
        self.QOS_PROFILE = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1
        )

        self.load_yaml()
        self._is_done = False

    @property
    def goal(self) -> Any:
        """Get the next goal to publish.
        
        Returns:
            Any: The next goal to publish.
        """
        return self._goal

    @property
    def is_done(self) -> bool:
        """Check if the formater is done.
        
        Returns:
            bool: True if the formater is done, False otherwise.
        """
        return self._is_done

    def load_yaml(self):
        """Load the YAML file with the goals."""
        print("Loading YAML file:", self._goals_file_path)
        with open(self._goals_file_path, "r") as file:
            self._yaml_file = yaml.safe_load(file)

    def process_yaml(self) -> None:
        """Process the YAML file with the goals."""
        raise NotImplementedError

    def log_publish(self) -> None:
        """Log the published goal."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset the formater to the initial state."""
        raise NotImplementedError
