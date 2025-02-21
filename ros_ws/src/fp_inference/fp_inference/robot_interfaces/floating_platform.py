from .base_robot_interface import BaseRobotInterface
from . import Registerable

from gymnasium import spaces
from std_msgs.msg import Int16MultiArray
import torch
import copy


class FloatingPlatformInterface(Registerable, BaseRobotInterface):
    """A class to interface with the floating platform.
    The interface is used to send commands to the platform and log the actions taken."""

    def __init__(self, *args, device: str | None = None, **kwargs):
        """Initialize the floating platform interface.

        Args:
            device: The device to perform computations on. Defaults to None."""

        super().__init__(*args, device=device, **kwargs)

        # Type of ROS message
        self.ROS_ACTION_TYPE = Int16MultiArray
        self.ROS_ACTION_QUEUE_SIZE = 1

        # Last actions is set to 0
        self._last_actions = torch.zeros((1, 8), device=self._device)
        self.commands = Int16MultiArray()
        actions = [0] * 9  # Everything off
        self.commands.data = actions

        # Action space
        self._action_space = spaces.MultiDiscrete([2] * 8)
        self._num_actions = 8
        
        self._remapping = [5, 1, 3, 7, 0, 4, 6, 2]

    @property
    def kill_action(self) -> Int16MultiArray:
        pre_kill_command = Int16MultiArray()
        actions = [1] + [0] * 8  # Leave bearigs on, thrusters off
        pre_kill_command.data = actions
        return pre_kill_command

    @property
    def pre_kill_action(self) -> Int16MultiArray:
        kill_command = Int16MultiArray()
        actions = [0] * 9  # Everything off
        kill_command.data = actions
        return kill_command
    
    def build_logs(self):
        """Build the logs for the robot interface. In this case, we log the thrusters firing."""

        super().build_logs()
        self._logs_specs["actions"] = [".t0", ".t1", ".t2", ".t3", ".t4", ".t5", ".t6", ".t7"]

    def cast_actions(self, actions) -> Int16MultiArray:
        # Actions are expected to be either 0 or 1
        super().cast_actions(actions)
        # Ensure actions are between 0 and 1
        actions = torch.clamp(actions, 0, 1)
        # Store the actions
        self._last_actions = copy.copy(actions)
        # Convert the actions to Int16MultiArray message
        actions = [1] + actions[0][self._remapping].int().tolist()
        self.commands.data = actions
        # Return the commands
        return self.commands

    def reset(self):
        """Reset the robot interface. This is called when the task is done and the robot needs to be reset for the next task."""

        super().reset()
        self._last_actions = torch.zeros((1, 8), device=self._device)
        actions = [0] * 9  # Everything off
        self.commands.data = actions
        self.build_logs()
