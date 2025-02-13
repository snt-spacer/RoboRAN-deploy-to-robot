#!/usr/bin/env python3

import rclpy
import rclpy.logging
from rclpy.node import Node
import rclpy.time
from std_msgs.msg import Bool
from typing import Tuple

from fp_inference.state_preprocessors import StatePreprocessorFactory
from fp_inference.observation_formaters import ObservationFormaterFactory
from fp_inference.robot_interfaces import RobotInterfaceFactory
from fp_inference.inference_runners import InferenceRunnerFactory 
from fp_inference.utils import Logger

from rcl_interfaces.msg import ParameterDescriptor


class RLTaskNode(Node):
    def __init__(self, task_name: str, goal: Tuple[float, float], num_steps_episode: int = 200, use_virtual_lab: bool = True, save_history: bool = False):
        super().__init__("rl_task_node")

        # Parameters
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

        # Build the task
        self.build()


    def build(self):
        """Build the task by creating the state preprocessor, observation formater, inference runner, and robot interface."""

        self.get_parameter('my_parameter').get_parameter_value()

        self.state_preprocessor = StatePreprocessorFactory.create(self._state_preprocessor_name, device=self._device)
        self.observation_formater = ObservationFormaterFactory.create(self._task_name, self.state_preprocessor, device=self._device)
        self.inference_runner = InferenceRunnerFactory.create(self._inference_runner_name, logdir=self._nn_log_dir, checkpoint_path=self._nn_checkpoint_path, device=self._device)
        self.robot_interface = RobotInterfaceFactory.create(self._robot_interface_name, device=self._device)

        logs_names = [self.state_preprocessor.logs_names, self.observation_formater.logs_names, self.robot_interface.logs_names]
        log_hooks = [self.state_preprocessor.logs, self.observation_formater.logs, self.robot_interface.logs]

        self.data_logger = Logger(logs_names, log_hooks, self._enable_logging)


        # ROS2 Subscriptions
        self.create_subscription(self.state_preprocessor.ROS_TYPE, "state_preprocessor_input", self.state_preprocessor.ROS_CALLBACK, self.state_preprocessor.ROS_QUEUE_SIZE)
        self.create_subscription(self.observation_formater.ROS_TYPE, "observation_formater_input", self.observation_formater.ROS_CALLBACK, self.state_preprocessor.ROS_QUEUE_SIZE)
        self.create_subscription(Bool, "reset_task", self.reset_task, 1)

        # ROS2 Publishers
        self.action_pub = self.create_publisher(self.robot_interface.ROS_ACTION_TYPE, "robot_interface_commands", self.robot_interface.ROS_ACTION_QUEUE_SIZE)

        # Timer for main task loop
        self.timer = self.create_timer(self._dt, self.run)

    def reset_task(self, msg: Bool):
        """Reset the task by resetting the state preprocessor, observation formater, inference runner, and robot interface."""

        self.get_logger().info("Reset received. Terminating task.")

        # Reset all the components
        self.state_preprocessor.reset()
        self.observation_formater.reset()
        self.inference_runner.reset()
        self.robot_interface.reset()

    def run_task(self):
        """Run the task by executing the main task loop."""
        self.get_logger().info("Running task...")
        # Create a rate object
        rate = self.create_rate(1.0/self._dt)

        # Main task loop
        while rclpy.ok():
            # Get the current observation
            self.observation_formater.format_observation(self.robot_interface.last_actions)
            # Get the action from the inference runner
            action = self.inference_runner.act(self.observation_formater.observation)
            # Publish the action
            self.action_pub(self.robot_interface.cast_actions(action))
            # Log the data
            self.data_logger.update()
            # Spin the node
            rclpy.spin_once(self)
            # Check if the task is completed
            if (self.observation_formater.task_completed) or (not self.observation_formater.task_is_live):
                break
            # Sleep
            rate.sleep()

    def run(self):
        # Check every second if a goal has been received
        wait_rate = self.create_rate(1.0)
        while rclpy.ok():
            # Wait for a goal to be received
            self.get_logger().info("Waiting for goal...")
            if self.observation_formater.task_is_live:
                self.get_logger().info("Goal received!")
                self.prime_state_preprocessor()
                self.run_task()
                self.get_logger().info("Task completed!")
                # Reset the robot interface
                self.robot_interface.reset()
                # Send immobilization command
                self.action_pub(self.robot_interface.kill_action)
                # Task is completed, save the logs.
                self.data_logger.save()
                # Returns if the task is completed
                if self._terminate_on_completion:
                    self.get_logger().info("In run once mode. Terminating node.")
                    break
                else:
                    self.get_logger().info("Resetting task.")
                    self.observation_formater.reset()
            wait_rate.sleep()
        
    def prime_state_preprocessor(self):
        """Wait for the state preprocesseor to be ready."""
        self.get_logger().info("Warming up the state preprocessor...")
        obs_rate = self.create_rate(1.0/self._dt)
        while rclpy.ok():
            if self.state_preprocessor.is_primed:
                break
            rclpy.spin_once(self)
            obs_rate.sleep()
        self.get_logger().info("State preprocessor is ready!")

    def shutdown(self):
        """Shutdown the node."""
        self.get_logger().info("Unexpected shutdown! Trying to kill the robot.")
        self.action_pub(self.robot_interface.kill_action)
        self.get_logger().info("Trying to save the logs.")
        self.data_logger.save()
        self.destroy_node()
        rclpy.shutdown()

    def clean_termination(self):
        self.destroy_node()
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)

    task_node = RLTaskNode()
    task_node.run()
    task_node.clean_termination()


if __name__ == "__main__":
    main()
