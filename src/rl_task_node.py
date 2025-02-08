#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import ByteMultiArray, Int16MultiArray
import numpy as np
import torch
import math
import yaml
import time
from typing import Tuple
from gymnasium import spaces
from skrl_inference.light_inference_runner import LightInferenceRunner
from skrl.utils.spaces.torch import unflatten_tensorized_space


class RLTaskNode(Node):
    def __init__(self, task_name: str, goal: Tuple[float, float], use_virtual_lab: bool = True):
        super().__init__("rl_task_node")

        # Mode Selection
        self.use_virtual_lab = use_virtual_lab
        self.topic_prefix = "/omniFPS/Robots/FloatingPlatform" if self.use_virtual_lab else "/vrpn_client_node/FP_exp_RL"

        # Task details
        self.task_name = task_name
        self.goal = torch.tensor(goal, dtype=torch.float32)
        self.goal_tolerance = 0.1
        self.steps_to_validate = 10

        # State representation
        self.prev_action = torch.zeros(8, dtype=torch.float32)
        self.state = torch.zeros(14, dtype=torch.float32)
        self.air_bearing = torch.tensor((1,), dtype=torch.float32)

        # Robot state variables
        self.robot_position = None
        self.robot_quat = None
        self.robot_vel = np.zeros(6)

        # Execution tracking
        self.current_step = 0
        self.successful_steps = 0
        self.num_steps_episode = 1000

        # Buffers for velocity computation
        self.pose_buffer = []
        self.time_buffer = []

        # ROS2 Subscriptions
        self.create_subscription(PoseStamped, f"{self.topic_prefix}/PoseStamped", self.pose_callback, 1)
        self.create_subscription(PoseStamped, "/FloatingPlatform/goal", self.goal_callback, 1)

        # ROS2 Publisher for Actions
        self.action_pub = self.create_publisher(ByteMultiArray if self.use_virtual_lab else Int16MultiArray, 
                                                f"{self.topic_prefix}/thrusters/input", 1)

        # Action Message Format
        self.thruster_msg = ByteMultiArray() if self.use_virtual_lab else Int16MultiArray()

        # Timer for main task loop
        self.timer = self.create_timer(0.1, self.task_loop)

        # Load RL model
        model_path = "models/FP_GoToPosition"
        self.load_model(model_path)

        self.get_logger().info(f"RL Task Node initialized for task: {self.task_name} (Virtual: {self.use_virtual_lab})")

    # --------------------- Utility Functions ---------------------

    def quaternion_to_heading(self, quat):
        """Convert a quaternion to a heading (yaw angle in radians)."""
        _, x, y, z = quat
        return math.atan2(2.0 * (x * y + z), 1.0 - 2.0 * (y * y + z * z))

    def get_rotation_matrix(self, heading: float) -> np.ndarray:
        """Generate a 2D rotation matrix for a given heading angle."""
        return np.array([
            [np.cos(heading), -np.sin(heading)],
            [np.sin(heading), np.cos(heading)]
        ])

    def get_angular_velocities(self, q: np.ndarray, dt: np.ndarray) -> np.ndarray:
        """Compute angular velocities from quaternions."""
        return (2 / dt) * np.array([
            q[:-1, 0] * q[1:, 1] - q[:-1, 1] * q[1:, 0] - q[:-1, 2] * q[1:, 3] + q[:-1, 3] * q[1:, 2],
            q[:-1, 0] * q[1:, 2] + q[:-1, 1] * q[1:, 3] - q[:-1, 2] * q[1:, 0] - q[:-1, 3] * q[1:, 1],
            q[:-1, 0] * q[1:, 3] - q[:-1, 1] * q[1:, 2] + q[:-1, 2] * q[1:, 1] - q[:-1, 3] * q[1:, 0]
        ])

    def derive_velocities(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute linear and angular velocities from pose history."""
        if len(self.pose_buffer) < 2:
            return np.zeros(3), np.zeros(3)

        dt = (self.time_buffer[-1] - self.time_buffer[0]) * 1e-9
        if dt == 0:
            return np.zeros(3), np.zeros(3)

        linear_positions = np.array([[pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]
                                     for pose in self.pose_buffer])
        linear_velocities = np.diff(linear_positions, axis=0) / (dt / len(self.pose_buffer))
        avg_linear_velocity = np.mean(linear_velocities, axis=0)

        quaternions = np.array([[pose.pose.orientation.w, pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z]
                                for pose in self.pose_buffer])
        angular_velocities = self.get_angular_velocities(quaternions, np.ones(len(quaternions) - 1) * dt / (len(quaternions) - 1))
        avg_angular_velocity = np.mean(angular_velocities, axis=1)

        return avg_linear_velocity, avg_angular_velocity

    # --------------------- Callbacks ---------------------

    def goal_callback(self, msg: PoseStamped):
        """Update the goal position from a received PoseStamped message."""
        self.goal = np.array([msg.pose.position.x, msg.pose.position.y])  # Only store x, y
        self.get_logger().info(f"Goal updated: {self.goal.tolist()}")


    def pose_callback(self, msg: PoseStamped):
        """Update the robot's pose, quaternion, and velocities from the pose message."""
        self.robot_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.robot_quat = np.array([msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z])

        # Maintain a fixed-size buffer for timestamps and poses
        self.pose_buffer.append(msg)
        self.time_buffer.append(self.get_clock().now().nanoseconds)
        
        if len(self.pose_buffer) > 30:
            self.pose_buffer.pop(0)
            self.time_buffer.pop(0)

        # Update robot velocities
        self.robot_vel[:3], self.robot_vel[3:] = self.derive_velocities()
        
    # --------------------- Core Logic ---------------------

    def turn_off_thrusters(self):
        """Turn off all thrusters with a delay before deactivating the air-bearing."""
        off_thrusters = [1] + [0] * 8  # Thrusters off, air-bearing on
        off_all = [0] * 9  # Everything off

        if self.use_virtual_lab:
            self.thruster_msg.data = [x.to_bytes(1, 'little') for x in off_thrusters]
        else:
            self.thruster_msg.data = off_thrusters

        self.action_pub.publish(self.thruster_msg)
        self.get_logger().info("Thrusters turned off.")

        time.sleep(1)  # Delay before turning off air-bearing

        if self.use_virtual_lab:
            self.thruster_msg.data = [x.to_bytes(1, 'little') for x in off_all]
        else:
            self.thruster_msg.data = off_all

        self.action_pub.publish(self.thruster_msg)
        self.get_logger().info("Air-bearing turned off.")

        
    def validate_task_completion(self) -> bool:
        """Check if the goal has been reached and maintained for the required steps."""
        if self.robot_position is None:
            return False

        # Compute distance to goal and update successful steps counter
        self.successful_steps = self.successful_steps + 1 if torch.norm(self.goal - self.robot_position[:2]) < self.goal_tolerance else 0

        return self.successful_steps >= self.steps_to_validate
  
    def load_model(self, log_dir: str):
        """Load the RL model from a given path."""
        env_params_path = f"{log_dir}/params/env.yaml"
        with open(env_params_path) as f:
            env_cfg = yaml.load(f, Loader=yaml.FullLoader)

        agent_params_path = f"{log_dir}/params/agent.yaml"
        with open(agent_params_path) as f:
            agent_params = yaml.safe_load(f)

        self.player = LightInferenceRunner(env_cfg, agent_params)
        self.player.build(f"{log_dir}/checkpoints/best_agent.pt")
        self.action_space = spaces.Tuple([spaces.Discrete(2)] * 8)

    def model_inference(self, state: torch.Tensor) -> torch.Tensor:
        """Perform model inference and return binary thruster actions."""
        with torch.inference_mode():
            actions = self.player.act(state, timestep=0, timesteps=0)[0]

        actions = torch.cat(unflatten_tensorized_space(self.action_space, actions)).squeeze()
        binary_actions = (actions > 0.5).int()

        return torch.cat([self.air_bearing, binary_actions], dim=0)

    def build_state(self) -> torch.Tensor:
        """Build the state tensor for RL inference."""
        if self.robot_position is None or self.robot_quat is None:
            return None

        position_error = self.goal - self.robot_position[:2]
        position_dist = torch.norm(torch.tensor(position_error, dtype=torch.float32))

        heading = self.quaternion_to_heading(self.robot_quat)
        target_heading = math.atan2(position_error[1], position_error[0])
        target_heading_error = math.atan2(math.sin(target_heading - heading), math.cos(target_heading - heading))

        R = self.get_rotation_matrix(-heading)
        lin_vel = torch.tensor(R @ self.robot_vel[:2], dtype=torch.float32)
        yaw_rate = torch.tensor(self.robot_vel[5], dtype=torch.float32)

        self.state = torch.cat([
            position_dist.unsqueeze(0),
            torch.tensor([math.cos(target_heading_error), math.sin(target_heading_error)]),
            lin_vel,
            yaw_rate.unsqueeze(0),
            self.prev_action
        ])

        return self.state

    def task_loop(self):
        """Main task loop executing RL inference and publishing actions."""
        self.build_state()
        if self.state is None:
            return

        action = self.model_inference(self.state)
        self.thruster_msg.data = [value.to_bytes(1, byteorder='little') for value in action.int().tolist()] if self.use_virtual_lab else action.int().tolist()
        self.action_pub.publish(self.thruster_msg)
        self.prev_action = action[1:]
        self.current_step += 1
        # Logs
        self.get_logger().info(f"Step: {self.current_step}, State: {self.state.tolist()}, Action: {action.tolist()}")
        # Check if goal is reached
        if self.validate_task_completion():
            self.get_logger().info(f"Task completed in {self.current_step} steps.")
            self.turn_off_thrusters()
            self.timer.cancel()
        if self.current_step >= self.num_steps_episode:
            self.get_logger().info(f"Task failed to complete in {self.num_steps_episode} steps.")
            self.turn_off_thrusters()
            self.timer.cancel()


def main(args=None):
    rclpy.init(args=args)

    # Example task: "go_to_position" with goal (5.0, 5.0)
    task_node = RLTaskNode("go_to_position", goal=(5.0, 5.0), use_virtual_lab=True)

    try:
        rclpy.spin(task_node)
    except KeyboardInterrupt:
        pass
    finally:
        task_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
