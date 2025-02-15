from gymnasium import spaces
from typing import Any
import torch
import copy


class BaseRobotInterface:
    """A base class for robot interfaces.
    The class is used to interface with a robot, send commands to the robot, and log the actions taken."""

    def __init__(self, device: str = "auto", **kwargs):
        """Initialize the robot interface."""

        # ROS parameters
        self.ROS_ACTION_TYPE = None
        self.ROS_ACTION_PUBLISHER = None
        self.ROS_ACTION_QUEUE_SIZE = None

        # Device selection
        if device == "auto":
            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        # Action space
        self._action_space: spaces = None

        # Lazy updates of state variables
        self._step_logs: int = 0

        # State priming
        self._step = 0

        # Log hook
        self.build_logs()

    def build_logs(self):
        """Build the logs for the robot interface. In this case, we log the thrusters firing."""

        # Log hook
        self._logs = {}
        self._logs_specs = {}
        self._logs["actions"] = None

    def update_logs(self):
        """Function used to update the logs for the robot interface."""

        self._logs["actions"] = self.last_actions

    @property
    def logs(self) -> dict[str, torch.Tensor]:
        """Return the logs for the robot interface.

        Returns:
            dict[str, torch.Tensor]: The logs for the robot interface."""

        if self._step_logs != self._step:
            self._step_logs = copy.copy(self._step)
            self.update_logs()
        return self._logs

    @property
    def logs_names(self) -> list[str]:
        """Return the logs names for the robot interface."""

        return self._logs.keys()

    @property
    def logs_specs(self):
        """Return the logs specifications for the robot interface."""

        return self._logs_specs

    @property
    def last_actions(self) -> torch.Tensor:
        """Return the last actions taken by the robot interface."""

        return self._last_actions

    @property
    def action_space(self) -> spaces.Space:
        """Return the action space for the robot interface."""

        return self._action_space

    @property
    def kill_action(self) -> Any:
        raise NotImplementedError("Kill action not implemented")

    def get_logs(self):
        """Hook function used by the logger to get the logs for the robot interface."""

        return self.logs

    def cast_actions(self, actions: torch.Tensor) -> Any:
        """Cast the actions to the robot interface format."""

        # Steps the interface when actions are casted
        self._step += 1

    def reset(self):
        """Reset the robot interface. This is called when the task is done and the robot needs to be reset for the next task."""

        self._step = 0
        self._step_logs = 0
        self.build_logs()
