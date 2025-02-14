from .base_robot_interface import BaseRobotInterface
from std_msgs.msg import Int16MultiArray
from gymnasium import spaces
import torch
import copy


class VirtualFloatingPlatformInterface(BaseRobotInterface):
    def __init__(self, *args, device: str | None = None, **kwargs):
        super().__init__(*args, device=device**kwargs)
        self.last_actions = torch.zeros((1, 8), device=self._device)
        self.commands = Int16MultiArray()
        self.commands.data = [0] * 9  # Everything off

        self.ROS_ACTION_TYPE = Int16MultiArray

        # Action space
        self._action_space = spaces.MultiDiscrete([2] * 8)

    @property
    def kill_action(self) -> Int16MultiArray:
        kill_command = Int16MultiArray()
        actions = [0] * 9
        kill_command.data = [value.to_bytes(1, byteorder="little") for value in actions.int().tolist()]
        return kill_command

    def cast_actions(self, actions) -> Int16MultiArray:
        # Actions are expected to be either 0 or 1
        super().cast_actions(actions)
        # Ensure actions are between 0 and 1
        actions = torch.clamp(actions, 0, 1)
        # Store the actions
        self.last_actions = copy.copy(actions)
        # Convert the actions to bytes message
        actions = [1] + actions.int().tolist()
        self.commands.data = [value.to_bytes(1, byteorder="little") for value in actions.int().tolist()]
        # Return the commands
        return self.commands

    def reset(self):
        super().reset()
        self.last_actions = torch.zeros((1, 8), device=self._device)
        actions = [0] * 9
        self.commands.data = [value.to_bytes(1, byteorder="little") for value in actions.int().tolist()]
        self.build_logs()
