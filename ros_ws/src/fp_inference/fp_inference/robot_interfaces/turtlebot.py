from .base_robot_interface import BaseRobotInterface
from . import Registerable

from gymnasium import spaces
from geometry_msgs.msg import Twist
import torch
import copy


class TurtlebotInterface(Registerable, BaseRobotInterface):
    """A class to interface with the turtlebot.
    The interface is used to send commands to the platform and log the actions taken."""

    def __init__(self, *args, device: str | None = None, **kwargs):
        """Initialize the turtlebot interface.

        Args:
            device: The device to perform computations on. Defaults to None."""

        super().__init__(*args, device=device, **kwargs)

        # Type of ROS message
        self.ROS_ACTION_TYPE = Twist

        # Last actions is set to 0
        self._last_actions = torch.zeros((1, 2), device=self._device)
        self.commands = Twist()
        actions = [0.0] * 2  # Everything off
        self.commands.linear.x = actions[0]
        self.commands.angular.z = actions[1]

        # Action space
        self._action_space = spaces.Box(low=-1, high=1, shape=(2,))
        self._num_actions = 2

    @property
    def kill_action(self) -> Twist:
        """Return the pre kill action for the robot interface. This is the action called before the kill action.
        This is meant to send an action that does not completely shuts down the robot, but preps it for the
        kill action."""

        kill_command = Twist()
        actions = [0.0] * 2  # Everything off
        self.commands.linear.x = actions[0]
        self.commands.angular.z = actions[1]
        return kill_command

    def build_logs(self) -> None:
        """Build the logs for the robot interface. In this case, we log the thrusters firing."""

        super().build_logs()
        self._logs_specs["actions"] = [".left", ".right"]

    def cast_actions(self, actions) -> Twist:
        """Cast the actions to the robot interface format.

        Args:
            actions (torch.Tensor): The actions to be casted into the robot interface format."""

        # Step the interface when actions are casted
        super().cast_actions(actions)

        # Actions are expected to be within -1 and 1, ensures actions are between -1 and 1
        actions = torch.clamp(actions, -1, 1)
        # Store the actions
        actions = torch.clamp(actions - self._last_actions, -0.2, 0.2) + self._last_actions
        self._last_actions = copy.copy(actions)
        
        # Convert the actions to bytes message
        actions = actions[0].tolist()
        self.commands.linear.x = actions[0] * 0.65
        self.commands.angular.z = actions[1] * 0.9

        print("Actions tb", self.commands)
        # Return the commands
        return self.commands

    def reset(self) -> None:
        """Reset the robot interface. This is called when the task is done and the robot needs to be reset for the
        next task."""

        super().reset()
        self._last_actions = torch.zeros((1, 2), device=self._device)

        # Kill everything on reset
        actions = [0.0] * 2  # Everything off
        self.commands.linear.x = actions[0]
        self.commands.angular.z = actions[1]
        self.build_logs()
