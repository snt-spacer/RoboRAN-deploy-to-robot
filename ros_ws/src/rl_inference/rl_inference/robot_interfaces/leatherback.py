from .base_robot_interface import BaseRobotInterface
from . import Registerable

from gymnasium import spaces
from ackermann_msgs.msg import AckermannDrive
import torch
import copy


class LeatherbackInterface(Registerable, BaseRobotInterface):
    """A class to interface with the leatherback.

    The interface is used to send commands to the leatherback and log the actions taken.
    """

    def __init__(self, *args, device: str | None = None, **kwargs) -> None:
        """Initialize the leatherback interface.

        Args:
            device: The device to perform computations on. Defaults to None.
        """
        super().__init__(*args, device=device, **kwargs)

        # Type of ROS message
        self.ROS_ACTION_TYPE = AckermannDrive

        # Last actions is set to 0
        self._last_actions = torch.zeros((1, 2), device=self._device)
        self._last_commands = torch.zeros((1, 2), device=self._device)
        self.commands = AckermannDrive()
        actions = [0.0] * 2  # Everything off
        self.commands.speed = actions[0]
        self.commands.steering_angle = actions[1]

        # Action space
        self._action_space = spaces.Box(low=-1, high=1, shape=(2,))
        self._num_actions = 2

    @property
    def kill_action(self) -> AckermannDrive:
        """Return the kill action for the robot interface.
        
        This is the action called to shut down the robot and prepping it for the next task.

        Returns:
            AckermannDrive: The kill action for the robot interface.
        """
        kill_command = AckermannDrive()
        actions = [0.0] * 2  # Everything off
        self.commands.speed = actions[0]
        self.commands.steering_angle = actions[1]
        return kill_command

    def build_logs(self) -> None:
        """Build the logs for the robot interface.
        
        In this case, we log the throttle and steering angle.
        """
        super().build_logs()
        self._logs_specs["actions"] = [".throttle", ".steering_angle"]
        self._logs_specs["commands"] = [".throttle.m/s", ".steering_angle.rad"]

    def cast_actions(self, actions) -> AckermannDrive:
        """Cast the actions to the robot interface format.

        Args:
            actions (torch.Tensor): The actions to be casted into the robot interface format.
        """
        # Step the interface when actions are casted
        super().cast_actions(actions)

        # Actions are expected to be within -1 and 1, ensures actions are between -1 and 1
        actions = torch.clamp(actions, -1, 1)

        # Store the actions
        actions = torch.clamp(actions - self._last_actions, -0.2, 0.2) + self._last_actions
        self._last_actions = copy.copy(actions)
        
        # Convert the actions to bytes message
        actions = actions[0].tolist()
        self.commands.speed = actions[0] * 2.0
        self.commands.steering_angle = actions[1] * 0.45
        self._last_commands[0,0] = self.commands.speed
        self._last_commands[0,1] = self.commands.steering_angle

        # Return the commands
        return self.commands

    def reset(self) -> None:
        """Reset the robot interface.
        
        This is called when the task is done and the robot needs to be reset for the next task.
        """
        super().reset()
        self._last_actions = torch.zeros((1, 2), device=self._device)
        self._last_commands = torch.zeros((1, 2), device=self._device)
        # Kill everything on reset
        actions = [0.0] * 2  # Everything off
        self.commands.speed = actions[0]
        self.commands.steering_angle = actions[1]
        self.build_logs()
