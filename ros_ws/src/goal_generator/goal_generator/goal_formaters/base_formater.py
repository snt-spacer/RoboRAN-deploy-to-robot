from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import yaml
import torch


class BaseFormaterCfg:
    pass


class BaseFormater:
    def __init__(
        self,
        goals_file_path: str,
        task_cfg: BaseFormaterCfg = BaseFormaterCfg(),
    ) -> None:

        self._task_cfg = task_cfg
        self._goals_file_path = goals_file_path
        self._goal = None

        self.ROS_TYPE = None
        self.QOS_PROFILE = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1
        )

        self.load_yaml()
        self._is_done = False

    @property
    def goal(self):
        return self._goal
    
    @property
    def is_done(self):
        return self._is_done

    def load_yaml(self):
        print("Loading YAML file:", self._goals_file_path)
        with open(self._goals_file_path, "r") as file:
            self._yaml_file = yaml.safe_load(file)

    def process_yaml(self):
        raise NotImplementedError

    def log_publish(self):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError