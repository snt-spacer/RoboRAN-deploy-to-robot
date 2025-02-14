from fp_inference.state_preprocessors import BaseStatePreProcessor
from dataclasses import dataclass
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

        # General parameters
        self._device = device
        self._state_preprocessor = state_preprocessor
        self.ROS_TYPE = None
        self.ROS_CALLBACK = self.update_goal_ROS
        self.ROS_QUEUE_SIZE = 1

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

        # Lazy updates
        self._step = 0
        self._step_logs = 0

        # Log hook
        self._logs = {}

        self.build_logs()

    def build_logs(self):
        self._logs = {}
        self._logs_specs = {}

    def update_logs(self):
        raise NotImplementedError("Update logs method not implemented")

    @property
    def task_is_live(self):
        return self._task_is_live

    @property
    def logs(self):
        # Update the logs if the step has changed
        if self._step_logs != self._step:
            self._step_logs = copy.copy(self._step)
            self.update_logs()
        return self._logs

    @property
    def logs_names(self):
        return self._logs.keys()

    @property
    def logs_specs(self):
        return self._logs_specs

    @property
    def observation(self):
        return self._observation

    @property
    def task_completed(self):
        return self._task_completed

    def get_logs(self):
        return self.logs

    def check_task_completion(self):
        raise NotImplementedError("Check task completion method not implemented")

    def format_observation(self, actions: torch.Tensor | None = None) -> None:
        if actions is None:
            raise ValueError("Actions must be provided to the observation formater.")

        self._step += 1

    def update_goal_ROS(self, **kwargs):
        raise NotImplementedError("Update goal ROS method not implemented")

    def reset(self, **kwargs):
        self.build_logs()
        self._task_completed = False
        self._task_is_live = False
        self._step = 0
        self._step_logs = 0
        self._observation = None
