from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from fp_inference.state_preprocessors import BaseStatePreProcessor
from dataclasses import dataclass
import gymnasium
import torch
import copy


class BaseFormaterCfg:
    pass


class BaseFormater:
    def __init__(
        self,
        state_preprocessor: BaseStatePreProcessor,
        device: str = "auto",
        max_steps: int = 500,
        task_cfg: BaseFormaterCfg = BaseFormaterCfg(),
        **kwargs,
    ) -> None:
        """Base class for task formatters. It defines the following:
        - The kind of ROS message used to update the goal for the task.
        - The observation space associated with the task.
        - The logs that are kept for the task.
        - The task to be done."""

        # General parameters
        self._device = device
        self._state_preprocessor = state_preprocessor
        self.ROS_TYPE = None
        self.ROS_CALLBACK = self.update_goal_ROS
        self.ROS_QOS_ = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=1
        )

        # Device selection
        if device == "auto":
            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        # Task parameters
        self._max_steps = max_steps
        self._task_cfg = task_cfg
        self._task_completed = False
        self._task_is_live = False
        self._observation = None
        self._observation_space = None

        # Lazy updates
        self._step = 0
        self._step_logs = 0

        # Log hook
        self._logs = {}

        self.build_logs()

    def build_logs(self) -> None:
        """Build the logs for the task. Variables to be logged should be implemented by the child class.
        Logs are used by the logger to automatically keep track of the task's progress. This provides a unified way to
        keep track of the task's progress and to compare different tasks and robots."""

        # Task logs
        self._logs = {}
        # Example: self._logs["target_position"] = torch.zeros((1, 2), device=self._device)
        # Note that the logs are stored as torch tensors, with the first dimension being the batch dimension.
        # Since we only control one robot at a time, the batch dimension is always 1.

        # Log specifications, this is used to provide a user friendly way to interpret the logs.
        self._logs_specs = {}
        # Example: self._logs_specs["target_position"] = [".x.m", ".y.m"]
        # Since `target_position` is a 2D tensor, we specify that it has two components, x and y. We also specify the
        # units of the components (meters in this case). Note that the separator "." is used to separate the variable
        # name from the component name and the units.

    def update_logs(self) -> None:
        """Update the logs for the task. This method is called every time the logs are accessed. It should be
        implemented by the child class. The logs should be updated based on the current state of the task.
        The logger performs a deep copy of the logs after this method is called, so the logs can be safely modified
        in place."""

        raise NotImplementedError("Update logs method not implemented")

    @property
    def observation_space(self) -> gymnasium.spaces.Space:
        """Property to get the observation space for the task."""
        return self._observation_space

    @property
    def task_is_live(self) -> bool:
        """Property indicating if the task is live. A task is live if it is being executed."""

        return self._task_is_live

    @property
    def logs(self) -> dict[str, torch.Tensor]:
        """Property to get the logs for the task. The logs are updated every time they are accessed.
        It uses a lazy update mechanism to avoid unnecessary computations."""

        # Update the logs if the step has changed
        if self._step_logs != self._step:
            self._step_logs = copy.copy(self._step)
            self.update_logs()
        return self._logs

    @property
    def logs_names(self) -> list[str]:
        """Property to get the names of the logs for the task."""

        return self._logs.keys()

    @property
    def logs_specs(self) -> dict[str, list[str]]:
        """Property to get the log specifications for the task."""

        return self._logs_specs

    @property
    def observation(self) -> torch.Tensor:
        """Property to get the observation for the task."""

        return self._observation

    @property
    def task_completed(self) -> bool:
        """Property indicating if the task has been completed."""

        return self._task_completed

    def get_logs(self) -> dict[str, torch.Tensor]:
        """Get the logs for the task. This method is used by the logger to get the logs for the task."""

        return self.logs

    def check_task_completion(self) -> None:
        """Check if the task has been completed. This method should be implemented by the child class.
        It performs the implace modification of the `_task_completed` attribute."""

        raise NotImplementedError("Check task completion method not implemented")

    def format_observation(self, actions: torch.Tensor | None = None) -> None:
        """Format the observation for the task. This method should be augmented by the child class.
        It performs the implace modification of the `_observation` attribute."""

        # Check if actions are provided
        if actions is None:
            raise ValueError("Actions must be provided to the observation formater.")

        # Update the step count
        self._step += 1

    def update_goal_ROS(self, **kwargs) -> None:
        """Update the goal using the ROS message. This method should be implemented by the child class."""

        raise NotImplementedError("Update goal ROS method not implemented")

    def reset(self, **kwargs) -> None:
        """Reset the task. This method should be augmented by the child class to reset the task-specific variables."""

        self.build_logs()
        self._task_completed = False
        self._task_is_live = False
        self._step = 0
        self._step_logs = 0
        self._observation = None
