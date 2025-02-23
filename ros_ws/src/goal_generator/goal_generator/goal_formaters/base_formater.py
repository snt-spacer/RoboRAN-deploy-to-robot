import yaml
import torch
from fp_inference.state_preprocessors import BaseStatePreProcessor


class BaseFormaterCfg:
    pass


class BaseFormater:
    def __init__(
        self,
        goals_file_path: str,
        state_preprocessor: BaseStatePreProcessor,
        task_cfg: BaseFormaterCfg = BaseFormaterCfg(),
    ) -> None:

        self._task_cfg = task_cfg
        self._goals_file_path = goals_file_path
        self._state_preprocessor = state_preprocessor
        self._goal = None
        self.task_completed = False
        self.send_goal = True

        self.load_yaml()

    @property
    def goal(self):
        return self._goal

    def load_yaml(self):
        print("yamma: ", self._goals_file_path)
        with open(self._goals_file_path, "r") as file:
            self._yaml_file = yaml.safe_load(file)

    def process_yaml(self):
        raise NotImplementedError

    def log_publish(self):
        raise NotImplementedError
