from gymnasium import spaces
from typing import Any
import torch
import copy

class BaseRobotInterface:
    def __init__(self):
        self.ROS_ACTION_TYPE = None
        self.ROS_ACTION_PUBLISHER = None
        self.ROS_ACTION_QUEUE_SIZE = None

        # Action space
        self._action_space: spaces = None

        # Lazy updates of state variables
        self._step_logs: int = 0

        # State priming
        self._step = 0

        # Log hook
        self.build_logs()

    def build_logs(self):
        # Log hook
        self._logs = {}
        self._logs["actions"] = None

    def update_logs(self):
        self._logs["actions"] = self.last_actions
    
    @property
    def logs(self) -> dict[str, torch.Tensor]: 
        if self._step_logs != self._step:
            self._step_logs = copy.copy(self._step)
            self.update_logs()
        return self._logs
    
    @property
    def logs_names(self) -> list[str]:
        return self._logs.keys()
    
    @property
    def last_actions(self) -> torch.Tensor:
        self._last_actions
    
    @property
    def action_space(self) -> spaces:
        return self._action_space
    
    @property
    def kill_action(self) -> Any:
        raise NotImplementedError("Kill action not implemented")

    def cast_actions(self, actions: torch.Tensor) -> Any:
        self._step += 1
    
    def reset(self):
        self._step = 0
        self._step_logs = 0
        self.build_logs()
    
