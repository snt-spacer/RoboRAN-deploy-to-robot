from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from .base_robot_interface import BaseRobotInterface
from . import Registerable

from gymnasium import spaces
from std_msgs.msg import Int16MultiArray
import torch
import copy


class FloatingPlatformInterface(Registerable, BaseRobotInterface):
    """A class to interface with the floating platform.
    The interface is used to send commands to the platform and log the actions taken."""

    def __init__(self, *args, device: str | None = None, **kwargs) -> None:
        """Initialize the floating platform interface.

        Args:
            device: The device to perform computations on. Defaults to None."""

        super().__init__(*args, device=device, **kwargs)

        # Type of ROS message
        self.ROS_ACTION_TYPE = Int16MultiArray
        self.QOS_PROFILE = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=1
        )

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
    def pre_kill_action(self) -> Int16MultiArray:
        """Return the kill action for the robot interface. This is the action called when the task is done.
        It is meant to stop the robot and prepping it for the next task."""

        kill_command = Int16MultiArray()
        actions = [0] * 9  # Everything off
        kill_command.data = actions
        return kill_command

    @property
    def kill_action(self) -> Int16MultiArray:
        """Return the pre kill action for the robot interface. This is the action called before the kill action.
        This is meant to send an action that does not completely shuts down the robot, but preps it for the
        kill action."""

        pre_kill_command = Int16MultiArray()
        actions = [1] + [0] * 8  # Leave bearigs on, thrusters off
        pre_kill_command.data = actions
        return pre_kill_command

    def build_logs(self) -> None:
        """Build the logs for the robot interface. In this case, we log the thrusters firing."""

        super().build_logs()
        self._logs_specs["actions"] = [".t0", ".t1", ".t2", ".t3", ".t4", ".t5", ".t6", ".t7"]

    def cast_actions(self, actions) -> Int16MultiArray:
        """Cast the actions to the robot interface format.

        Args:
            actions (torch.Tensor): The actions to be casted into the robot interface format."""

        # Step the interface when actions are casted
        super().cast_actions(actions)

        # Actions are expected to be either 0 or 1, ensures actions are between 0 and 1
        actions = torch.clamp(actions, 0, 1)
        # Store the actions
        self._last_actions = copy.copy(actions)
        # Convert the actions to Int16MultiArray message
        actions = [1] + actions[0][self._remapping].int().tolist()
        self.commands.data = actions
        # Return the commands
        return self.commands

    def reset(self) -> None:
        """Reset the robot interface. This is called when the task is done and the robot needs to be reset for the
        next task."""

        super().reset()
        self._last_actions = torch.zeros((1, 8), device=self._device)

        # Kill everything on reset
        actions = [0] * 9  # Everything off
        self.commands.data = actions
        self.build_logs()
