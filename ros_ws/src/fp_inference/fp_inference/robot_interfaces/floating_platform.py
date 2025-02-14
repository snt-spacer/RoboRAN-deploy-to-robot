from .base_robot_interface import BaseRobotInterface
from . import Registerable

from gymnasium import spaces
from std_msgs.msg import ByteMultiArray
import torch
import copy


class FloatingPlatformInterface(Registerable, BaseRobotInterface):
    def __init__(self, *args, device: str | None = None, **kwargs):
        super().__init__(*args, device=device, **kwargs)

        # Type of ROS message
        self.ROS_ACTION_TYPE = ByteMultiArray
        self.ROS_ACTION_QUEUE_SIZE = 1

        # Last actions is set to 0
        self._last_actions = torch.zeros((1, 8), device=self._device)
        self.commands = ByteMultiArray()
        actions = [0] * 9  # Everything off
        self.commands.data = [value.to_bytes(1, byteorder="little") for value in actions]

        # Action space
        self._action_space = spaces.MultiDiscrete([2] * 8)

    @property
    def kill_action(self) -> ByteMultiArray:
        kill_command = ByteMultiArray()
        actions = [0] * 9  # Everything off
        kill_command.data = [value.to_bytes(1, byteorder="little") for value in actions]
        return kill_command

    def cast_actions(self, actions) -> ByteMultiArray:
        # Actions are expected to be either 0 or 1
        super().cast_actions(actions)
        # Ensure actions are between 0 and 1
        actions = torch.clamp(actions, 0, 1)
        # Store the actions
        self._last_actions = copy.copy(actions)
        # Convert the actions to bytes message
        actions = [1] + actions[0].int().tolist()
        self.commands.data = [value.to_bytes(1, byteorder="little") for value in actions]
        # Return the commands
        return self.commands

    def reset(self):
        super().reset()
        self._last_actions = torch.zeros((1, 8), device=self._device)
        actions = [0] * 9  # Everything off
        self.commands.data = [value.to_bytes(1, byteorder="little") for value in actions]
        self.build_logs()
