#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import ByteMultiArray, Int16MultiArray
import numpy as np
import torch
import math
import time
from typing import Tuple
from skrl_inference.light_inference_runner import LightInferenceRunner
from skrl.utils.spaces.torch import unflatten_tensorized_space
from .trajectory_plotter import plot_episode_data

from fp_inference.state_preprocessors import StatePreprocessorFactory
from fp_inference.observation_formaters import ObservationFormaterFactory
from fp_inference.utils import Logger

from rcl_interfaces.msg import ParameterDescriptor


class RLTaskNode(Node):
    def __init__(self, task_name: str, goal: Tuple[float, float], num_steps_episode: int = 200, use_virtual_lab: bool = True, save_history: bool = False):
        super().__init__("rl_task_node")

        # Parameters
        task_name_desc = ParameterDescriptor(description='The name of the task to be executed. Currently the following tasks are supported: {}".'.format(", ".join(ObservationFormaterFactory.registry.keys())))
        self.declare_parameter('task_name', 'GoToPosition', task_name_desc)
        state_preprocessor_name_desc = ParameterDescriptor(description='The name of the state preprocessor to be used. Currently the following state preprocessors are supported: {}".'.format(", ".join(StatePreprocessorFactory.registry.keys())))
        self.declare_parameter('state_preprocessor_name', 'Optitrack', state_preprocessor_name_desc)
        robot_interface_name_desc = ParameterDescriptor(description='The name of the robot interface to be used. Currently the following robot interfaces are supported: {}".'.format(", ".join(RobotInterfaceFactory.registry.keys())))
        self.declare_parameter('robot_interface_name', 'FloatingPlatform', robot_interface_name_desc)
        inference_runner_name_desc = ParameterDescriptor(description='The name of the inference runner to be used. Currently the following inference runners are supported: {}".'.format(", ".join(InferenceRunnerFactory.registry.keys())))
        self.declare_parameter('inference_runner_name', 'SKRLInferenceRunner', inference_runner_name_desc)
        enable_logging_desc = ParameterDescriptor(description='Enable logging of the task execution.')
        self.declare_parameter('enable_logging', False, enable_logging_desc)
        device_desc = ParameterDescriptor(description='The device to be used for the task. If set to "auto", the device will be selected automatically.')
        self.declare_parameter('device', 'auto', device_desc)
        max_steps_desc = ParameterDescriptor(description='The number of steps to be executed in each episode.')
        self.declare_parameter('max_steps', 200, max_steps_desc)
        dt_desc = ParameterDescriptor(description='The time step with which the task will be executed.')
        self.declare_parameter('dt', 1/15.0, dt_desc)
        nn_log_dir_desc = ParameterDescriptor(description='The directory where the neural network model is stored.')
        self.declare_parameter('nn_log_dir', None, nn_log_dir_desc)
        nn_checkpoint_path_desc = ParameterDescriptor(description='The path to the neural network model checkpoint.')
        self.declare_parameter('nn_checkpoint_path', None, nn_checkpoint_path_desc)

        self.build()

        # ROS2 Subscriptions
        self.create_subscription(self.state_preprocessor.ROS_TYPE, "/state_preprocessor_input", self.state_preprocessor.ROS_CALLBACK, self.state_preprocessor.ROS_QUEUE_SIZE)
        self.create_subscription(self.observation_formater.ROS_TYPE, "/observation_formater_input", self.observation_formater.ROS_CALLBACK, self.state_preprocessor.ROS_QUEUE_SIZE)

        # ROS2 Publisher for Actions
        self.action_pub = self.create_publisher(ByteMultiArray if self.use_virtual_lab else Int16MultiArray, f"{self.topic_prefix}/thrusters/input", 1)

        # Timer for main task loop
        self.timer = self.create_timer(0.1, self.run)

    def build(self):
        self.state_preprocessor = StatePreprocessorFactory.create(self.state_preprocessor_name, device=self.device)
        self.observation_formater = ObservationFormaterFactory.create(self.task_name, self.state_preprocessor, device=self.device)
        self.inference_runner = InferenceRunnerFactory.create(self.inference_runner_name, logdir=self.nn_log_dir, checkpoint_path=self.nn_checkpoint_path, device=self.device)
        self.robot_interface = RobotInterfaceFactory.create(self.robot_interface_name, device=self.device)

        logs_names = [self.state_preprocessor.logs_names, self.observation_formater.logs_names, self.robot_interface.logs_names]
        log_hooks = [self.state_preprocessor.logs, self.observation_formater.logs, self.robot_interface.logs]

        self.logger = Logger(logs_names, log_hooks, self.enable_logging)

    def run(self):
        # Get the current observation
        observation = self.observation_formater.observation

        # Get the action from the inference runner
        action = self.inference_runner.get_action(observation)

        # Publish the action
        self.publish_action(action)

        # Log the data
        self.logger.log()
        


def main(args=None):
    rclpy.init(args=args)

    # Example task: "go_to_position" with goal (5.0, 5.0)
    task_node = RLTaskNode("go_to_position", goal=(5.0, 5.0), num_steps_episode=50, use_virtual_lab=False, save_history=True)

    try:
        rclpy.spin(task_node)
    except KeyboardInterrupt:
        pass
    finally:
        task_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
