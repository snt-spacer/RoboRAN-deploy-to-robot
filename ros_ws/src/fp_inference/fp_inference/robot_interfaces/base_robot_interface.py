from gymnasium import spaces
from typing import Any
import torch
import copy


class BaseRobotInterface:
    """A base class for robot interfaces.
    The class is used to interface with a robot, send commands to the robot, and log the actions taken."""

    def __init__(self, device: str = "auto", **kwargs) -> None:
        """Initialize the robot interface.

        Args:
            device (str): The device to perform computations on. Defaults to "auto"."""

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
        self._last_actions: torch.Tensor = None
        self._num_actions: int = 0

        # Lazy updates of state variables
        self._step_logs: int = 0

        # State priming
        self._step = 0

        # Log hook
        self.build_logs()

    def build_logs(self) -> None:
        """Build the logs for the robot interface. Logging robot related things, such as actions."""

        # Robot logs
        self._logs = {}

        # Log specifications, this is used to provide a user friendly way to interpret the logs.
        self._logs_specs = {}
        self._logs["actions"] = None

    def update_logs(self) -> None:
        """Function used to update the logs for the robot interface."""

        self._logs["actions"] = self.last_actions

    @property
    def num_actions(self) -> int:
        """Return the number of actions for the robot interface."""

        return self._num_actions

    @property
    def logs(self) -> dict[str, torch.Tensor]:
        """Return the logs for the robot interface."""

        if self._step_logs != self._step:
            self._step_logs = copy.copy(self._step)
            self.update_logs()
        return self._logs

    @property
    def logs_names(self) -> list[str]:
        """Return the logs names for the robot interface."""

        return self._logs.keys()

    @property
    def logs_specs(self) -> dict[str, list[str]]:
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
    def pre_kill_action(self) -> Any:
        """Return the pre kill action for the robot interface. This is the action called before the kill action.
        This is meant to send an action that does not completely shuts down the robot, but preps it for the
        kill action."""

        # In most cases this is not needed and is set to the regular kill action.
        return self.kill_action

    @property
    def kill_action(self) -> Any:
        """Return the kill action for the robot interface. This is the action called when the task is done.
        It is meant to stop the robot and prepping it for the next task."""

        raise NotImplementedError("Kill action not implemented")

    def get_logs(self) -> dict[str, torch.Tensor]:
        """Hook function used by the logger to get the logs for the robot interface."""

        return self.logs

    def cast_actions(self, actions: torch.Tensor) -> Any:
        """Cast the actions to the robot interface format.

        Args:
            actions (torch.Tensor): The actions to be casted into the robot interface format."""

        # Steps the interface when actions are casted
        self._step += 1

    def reset(self) -> None:
        """Reset the robot interface. This is called when the task is done and the robot needs to be reset for the
        next task."""

        self._step = 0
        self._step_logs = 0
        self.build_logs()
