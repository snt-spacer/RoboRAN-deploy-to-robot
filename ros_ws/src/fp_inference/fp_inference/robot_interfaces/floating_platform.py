from .base_robot_interface import BaseRobotInterface

from std_msgs.msg import ByteMultiArray
import torch
import copy


class FloatingPlatformInterface(BaseRobotInterface):
    def __init__(self, *args, device: str | None = None, **kwargs):
        super().__init__(*args, device=device, **kwargs)
        self.last_actions = torch.zeros((1, 8), device=self._device)
        self.commands = ByteMultiArray()
        self.commands.data = [0] * 9  # Everything off

        self.ROS_ACTION_TYPE = ByteMultiArray

    @property
    def kill_action(self) -> ByteMultiArray:
        kill_command = ByteMultiArray()
        kill_command.data = [0] * 9
        return kill_command

    def cast_actions(self, actions) -> ByteMultiArray:
        # Actions are expected to be either 0 or 1
        super().cast_actions(actions)
        # Ensure actions are between 0 and 1
        actions = torch.clamp(actions, 0, 1)
        # Store the actions
        self.last_actions = copy.copy(actions)
        # Convert the actions to bytes message
        self.commands.data = [1] + actions.int().tolist()
        # Return the commands
        return self.commands

    def reset(self):
        super().reset()
        self.last_actions = torch.zeros((1, 8), device=self._device)
        self.commands.data = [0] * 9  # Everything off
        self.build_logs()
