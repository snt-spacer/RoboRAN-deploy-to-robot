from fp_inference.state_preprocessors import BaseStatePreProcessor
import torch
import copy

class BaseFormater:
    def __init__(self, state_preprocessor: BaseStatePreProcessor, device: str = 'cuda', **kwargs):

        # General parameters
        self._device = device
        self._state_preprocessor = state_preprocessor
        self.ROS_TYPE = None
        self.ROS_CALLBACK = self.update_goal_ROS
        self.ROS_QUEUE_SIZE = 1

        # Lazy goal update
        self._observation_step = 0
        self._step = 0
        self._last_preprocessor_step = 0
        self._step_logs = 0

        # Log hook
        self._logs = {}

    def build_logs(self):
        self._logs = {}

    def update_logs(self):
        raise NotImplementedError("Update logs method not implemented")

    @property
    def logs(self):
        if self._step_logs != self._step:
            self._step_logs = copy.copy(self._step)
            self.update_logs()
        return self._logs
    
    @property
    def logs_names(self):
        return self._logs.keys()

    @property
    def observation(self):
        if (self._observation_step != self._step) or (self._state_preprocessor.step != self._last_preprocessor_step):
            self._observation_step = copy.copy(self._step)
            self._last_preprocessor_step = copy.copy(self._state_preprocessor.step)
        return self.get_observation()

    def get_observation(self, action: torch.Tensor | None = None):
        raise NotImplementedError("Get observation method not implemented")

    def update_goal(self, **kwargs):
        raise NotImplementedError("Update goal method not implemented")

    def update_goal_ROS(self, **kwargs):
        raise NotImplementedError("Update goal ROS method not implemented")

    def reset(self, **kwargs):
        raise NotImplementedError("Reset method not implemented")
    
