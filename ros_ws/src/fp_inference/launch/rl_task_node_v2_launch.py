from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='fp_inference',
            executable='rl_task_node_v2',
            name='rl_task_node_v2',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'task_id': 0}
        task_name_desc = ParameterDescriptor(description='The name of the task to be executed. Currently the following tasks are supported: {}".'.format(", ".join(ObservationFormaterFactory.registry.keys())))
        self.declare_parameter('task_name', 'GoToPosition', task_name_desc)
        self._task_name = self.get_parameter('task_name').get_parameter_value().string_value
        state_preprocessor_name_desc = ParameterDescriptor(description='The name of the state preprocessor to be used. Currently the following state preprocessors are supported: {}".'.format(", ".join(StatePreprocessorFactory.registry.keys())))
        self.declare_parameter('state_preprocessor_name', 'Optitrack', state_preprocessor_name_desc)
        self._state_preprocessor_name = self.get_parameter('state_preprocessor_name').get_parameter_value().string_value
        robot_interface_name_desc = ParameterDescriptor(description='The name of the robot interface to be used. Currently the following robot interfaces are supported: {}".'.format(", ".join(RobotInterfaceFactory.registry.keys())))
        self.declare_parameter('robot_interface_name', 'FloatingPlatform', robot_interface_name_desc)
        self._robot_interface_name = self.get_parameter('robot_interface_name').get_parameter_value().string_value
        inference_runner_name_desc = ParameterDescriptor(description='The name of the inference runner to be used. Currently the following inference runners are supported: {}".'.format(", ".join(InferenceRunnerFactory.registry.keys())))
        self.declare_parameter('inference_runner_name', 'SKRLInferenceRunner', inference_runner_name_desc)
        self._inference_runner_name = self.get_parameter('inference_runner_name').get_parameter_value().string_value
        enable_logging_desc = ParameterDescriptor(description='Enable logging of the task execution.')
        self.declare_parameter('enable_logging', False, enable_logging_desc)
        self._enable_logging = self.get_parameter('enable_logging').get_parameter_value().bool_value
        device_desc = ParameterDescriptor(description='The device to be used for the task. If set to "auto", the device will be selected automatically.')
        self.declare_parameter('device', 'auto', device_desc)
        self._device = self.get_parameter('device').get_parameter_value().string_value
        max_steps_desc = ParameterDescriptor(description='The number of steps to be executed in each episode.')
        self.declare_parameter('max_steps', 200, max_steps_desc)
        self._max_steps = self.get_parameter('max_steps').get_parameter_value().integer_value
        dt_desc = ParameterDescriptor(description='The time step with which the task will be executed.')
        self.declare_parameter('dt', 1/15.0, dt_desc)
        self._dt = self.get_parameter('dt').get_parameter_value().double_value
        nn_log_dir_desc = ParameterDescriptor(description='The directory where the neural network model is stored.')
        self.declare_parameter('nn_log_dir', None, nn_log_dir_desc)
        self._nn_log_dir = self.get_parameter('nn_log_dir').get_parameter_value().string_value
        nn_checkpoint_path_desc = ParameterDescriptor(description='The path to the neural network model checkpoint.')
        self.declare_parameter('nn_checkpoint_path', None, nn_checkpoint_path_desc)
        self._nn_checkpoint_path = self.get_parameter('nn_checkpoint_path').get_parameter_value().string_value
        terminate_on_completion_desc = ParameterDescriptor(description='Terminate the node when the goal is reached.')
        self.declare_parameter('terminate_on_completion', False, terminate_on_completion_desc)
        self._terminate_on_completion = self.get_parameter('terminate_on_completion').get_parameter_value().bool_value
        logs_save_path_desc = ParameterDescriptor(description='The path where the logs will be saved.')
        self.declare_parameter('logs_save_path', None, logs_save_path_desc)
        self._logs_save_path = self.get_parameter('logs_save_path').get_parameter_value().string_value
                ]
        )
    ])