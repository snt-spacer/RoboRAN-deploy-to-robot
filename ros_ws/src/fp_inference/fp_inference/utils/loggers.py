class Logger:
    def __init__(self, logs_names:list=[], hooks: list = [], enabled: bool = False):
        self.logs_names = logs_names
        self.hooks = hooks
        self.enabled = enabled

        self.build_logs()

    @property
    def logs(self):
        return self._logs
    
    @property
    def logs_names(self):
        return self._logs.keys()

    def build_logs(self):
        self._logs = {}
        for logs_names in self.logs_names:
            for log_name in logs_names:
                # Check that the log name is not already in the logs
                if log_name not in self._logs:
                    raise ValueError(f"Log name {log_name} already exists.")
                # Add the log name to the logs
                self._logs[log_name] = []

    def collect_logs(self, logs: dict):
        for hook in self.hooks:
            logs = hook()
            for key, value in logs.items():
                self._logs[key].append(value)

    def update(self):
        if self.enabled:
            self.collect_logs()