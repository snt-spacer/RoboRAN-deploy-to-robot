from .base_robot_interface import BaseRobotInterface
from . import Registerable

from std_msgs.msg import ByteMultiArray
from gymnasium import spaces
import torch
import copy


class VirtualFloatingPlatformInterface(Registerable, BaseRobotInterface):
    """A class to interface with the virtual floating platform.
    The interface is used to send commands to the platform and log the actions taken."""

    def __init__(self, *args, device: str | None = None, **kwargs):
        """Initialize the virtual floating platform interface.

        Args:
            device: The device to perform computations on. Defaults to None."""

        super().__init__(*args, device=device, **kwargs)

        # Type of ROS message
        self.ROS_ACTION_TYPE = ByteMultiArray
        self.ROS_ACTION_QUEUE_SIZE = 1

        # Last actions is set to 0
        self._last_actions = torch.zeros((1, 8), device=self._device)
        self.commands = ByteMultiArray()
        actions = [0] * 9  # Everything off
        self.commands.data = [value.to_bytes(1, byteorder="little") for value in actions]

        self._remapping = [7, 0, 1, 2, 3, 4, 5, 6]

        # Action space
        self._action_space = spaces.MultiDiscrete([2] * 8)
        self._num_actions = 8

    @property
    def kill_action(self) -> ByteMultiArray:
        """Return the kill action for the robot interface. This is the action called when the task is done.
        It is meant to stop the robot and prepping it for the next task.

        Returns:
            ByteMultiArray: The kill action for the robot interface."""

        kill_command = ByteMultiArray()
        actions = [0] * 9
        kill_command.data = [value.to_bytes(1, byteorder="little") for value in actions]
        return kill_command

    def build_logs(self):
        """Build the logs for the robot interface. In this case, we log the thrusters firing."""

        super().build_logs()
        self._logs_specs["actions"] = [".t0", ".t1", ".t2", ".t3", ".t4", ".t5", ".t6", ".t7"]

    def cast_actions(self, actions: torch.Tensor) -> ByteMultiArray:
        """Cast the actions to the robot interface format.

        Args:
            actions (torch.Tensor): The actions to be casted.

        Returns:
            ByteMultiArray: The actions in the robot interface format."""

        # Actions are expected to be either 0 or 1
        super().cast_actions(actions)
        # Ensure actions are between 0 and 1
        actions = torch.clamp(actions, 0, 1)
        # Store the actions
        self._last_actions = copy.copy(actions)
        # Convert the actions to bytes message
        actions = [1] + actions[0][self._remapping].int().tolist()
        self.commands.data = [value.to_bytes(1, byteorder="little") for value in actions]
        # Return the commands
        return self.commands

    def reset(self):
        """Reset the robot interface. This is called when the task is done and the robot needs to be reset for the next task."""
        super().reset()
        self._last_actions = torch.zeros((1, 8), device=self._device)
        actions = [0] * 9
        self.commands.data = [value.to_bytes(1, byteorder="little") for value in actions]
        self.build_logs()
