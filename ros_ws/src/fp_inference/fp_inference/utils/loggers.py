import pandas as pd
import numpy as np
import datetime
import copy
import torch
import os


# Could be nicer, if the subclasses where accessing an instance of the logger.
class Logger:
    def __init__(self, objects: list = [], enabled: bool = False, save_path: str | None = None):
        self._enabled = enabled
        self._objects = objects
        self.initialize()

        if save_path is None:
            raise ValueError("Save path is not defined.")
        self._save_path = save_path

        self.build_logs()
        self.print_info()

    @property
    def logs(self) -> dict[str, list]:
        return self._logs

    @property
    def logs_names(self) -> list[str]:
        return self._logs.keys()

    def initialize(self):
        self._logs_names = [o.logs_names for o in self._objects]
        logs_specs = [o.logs_specs for o in self._objects]
        self._hooks = [o.get_logs for o in self._objects]
        # Flatten the specs
        self._logs_specs = {}
        for logs_spec in logs_specs:
            self._logs_specs.update(logs_spec)

    def print_info(self) -> None:
        print("=============================================")
        print("Logger information:")
        print(f"Enabled: {self._enabled}")
        print(f"Save path: {self._save_path}")
        print(f"Registered {len(self._hooks)} logging hooks.")
        print(f"Logging the following variables:")
        [print(f" +{log_name}") for log_name in self.logs_names]
        print("=============================================")

    def build_logs(self) -> None:
        self._logs = {}
        for logs_names in self._logs_names:
            for log_name in logs_names:
                # Check that the log name is not already in the logs
                if log_name in self._logs:
                    raise ValueError(f"Log name {log_name} already exists.")
                # Add the log name to the logs
                self._logs[log_name] = []

    def collect_logs(self) -> None:
        """Collect logs from the hooks. The logs contain torch.Tensors."""

        for hook in self._hooks:
            logs = hook()
            for key, value in logs.items():
                self._logs[key].append(copy.deepcopy(value))

    def update(self) -> None:
        if self._enabled:
            self.collect_logs()

    def convert_buffer(self) -> pd.DataFrame:
        data = {}
        for name, list_tensor_log in self._logs.items():
            tensor_log = torch.stack(list_tensor_log).squeeze(1)
            numpy_log = tensor_log.cpu().numpy()
            specs = self._logs_specs[name]
            for i, spec in enumerate(specs):
                data[name + spec] = numpy_log[:, i]
        df = pd.DataFrame(data)
        return df

    def save(self, robot_interface_name: str, inference_runner_name: str, observation_formater_name: str) -> None:
        if self._enabled:
            date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            name = f"{inference_runner_name}_{robot_interface_name}_{observation_formater_name}_{date}"
            os.makedirs(self._save_path, exist_ok=True)
            os.makedirs(os.path.join(self._save_path, "logs"), exist_ok=True)
            save_path = os.path.join(self._save_path, "logs", name)
            df = self.convert_buffer()
            df.to_csv(save_path + ".csv", sep=",", index=False, header=True)
