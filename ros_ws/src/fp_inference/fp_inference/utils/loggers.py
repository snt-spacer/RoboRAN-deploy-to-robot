import datetime
import os


class Logger:
    def __init__(self, logs_names: list = [], hooks: list = [], enabled: bool = False, save_path: str | None = None):
        self._logs_names = logs_names
        self.hooks = hooks

        self.enabled = enabled

        if save_path is None:
            raise ValueError("Save path is not defined.")
        self.save_path = save_path

        self.build_logs()
        self.print_info()

    @property
    def logs(self) -> dict[str, list]:
        return self._logs

    @property
    def logs_names(self) -> list[str]:
        return self._logs.keys()
    
    def print_info(self) -> None:
        print("=============================================")
        print("Logger information:")
        print(f"Enabled: {self.enabled}")
        print(f"Save path: {self.save_path}")
        print(f"Loggings:")
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
        """Collect logs from the hooks. The logs contain torch.Tensors.

        Args:
            logs (dict): The logs to be collected."""

        for hook in self.hooks:
            logs = hook()
            for key, value in logs.items():
                self._logs[key].append(value)

    def update(self) -> None:
        if self.enabled:
            self.collect_logs()

    def save(self, robot_interface_name: str, inference_runner_name: str, observation_formater_name: str) -> None:
        if self.enabled:
            date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            name = f"{inference_runner_name}_{robot_interface_name}_{observation_formater_name}_{date}"
            os.makedirs(self.save_path, exist_ok=True)
            os.makedirs(os.path.join(self.save_path, "logs"), exist_ok=True)

            #raise NotImplementedError("Save method not implemented.")
