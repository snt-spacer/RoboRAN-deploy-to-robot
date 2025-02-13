from gymnasium import spaces
import torch
import copy

class BaseRobotInterface:
    def __init__(self):
        self.ROS_ACTION_TYPE = None
        self.ROS_ACTION_PUBLISHER = None
        self.ROS_ACTION_QUEUE_SIZE = None

        # Action space
        self._action_space: spaces = None
        self._action: torch.Tensor = None

        # Lazy updates of state variables
        self._step_logs: int = 0

        # State priming
        self._step = 0

        # Log hook
        self.build_logs()

    def build_logs(self):
        # Log hook
        self._logs = {}

    def update_logs(self):
        raise NotImplementedError("update_logs method must be implemented in the child class.")
    
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
        raise NotImplementedError("last_actions property must be implemented in the child class.")
    
    @property
    def action_space(self) -> spaces:
        return self._action_space
    
    def cast_actions(self, actions: torch.Tensor):
        raise NotImplementedError("cast_actions method must be implemented in the child class.")
    
    def reset(self):
        self._step = 0
        self._step_logs = 0
        self.build_logs()
    
