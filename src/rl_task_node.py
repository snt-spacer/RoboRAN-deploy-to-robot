#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import numpy as np
import torch
from typing import Tuple


class RLTaskNode(Node):
    def __init__(self, task_name: str, goal: Tuple[float, float]):
        super().__init__("rl_task_node")

        # Task details
        self.task_name = task_name
        self.goal = torch.tensor(goal, dtype=torch.float32)
        self.goal_tolerance = 0.1  # Distance tolerance for goal completion
        self.steps_to_validate = 10  # Steps to maintain the goal to validate success

        # State representation
        self.prev_action = torch.zeros(9, dtype=torch.float32)  # Shape (9,) [0: airbearing, 1-8: thrusters]
        self.state = torch.zeros(15, dtype=torch.float32)  # Shape (15,) [0: target_dist, 1-2: target_heading, 3-4: lin_vel, 5: yaw_rate, 6-15: prev_action]
        self.air_bearing = torch.tensor((0,), dtype=torch.float32)  # Placeholder for air bearing value    

        # Robot state variables
        self.robot_position = None
        self.robot_quat = None
        self.robot_vel = np.zeros(6)  # Linear (x, y, z) + Angular (roll, pitch, yaw)

        # Task execution tracking
        self.current_step = 0
        self.successful_steps = 0

        # Buffers for velocity computation
        self.pose_buffer = []
        self.time_buffer = []

        # ROS2 communication
        self.create_subscription(PoseStamped, "/vrpn_client_node/FP_exp_RL/pose", self.pose_callback, 10)

        # Timer for repeating task loop (10 Hz)
        self.timer = self.create_timer(0.1, self.task_loop)

        self.get_logger().info(f"RL Task Node initialized for task: {self.task_name}")

    def pose_callback(self, msg: PoseStamped):
        """Callback for OptiTrack pose messages."""
        # Update pose and quaternion
        pos = msg.pose.position
        quat = msg.pose.orientation
        self.robot_position = np.array([pos.x, pos.y, pos.z])
        self.robot_quat = np.array([quat.w, quat.x, quat.y, quat.z])

        # Store pose and timestamp in buffers
        self.pose_buffer.append(msg)
        self.time_buffer.append(self.get_clock().now().nanoseconds)
        if len(self.pose_buffer) > 30:  # Limit buffer size
            self.pose_buffer.pop(0)
            self.time_buffer.pop(0)

        # Update linear and angular velocities
        self.robot_vel[:3], self.robot_vel[3:] = self.derive_velocities()

    def quaternion_to_heading(self, quat):
        """
        Converts a quaternion to a heading (yaw angle in radians).
        """
        _, x, y, z = quat
        heading = np.arctan2(2.0 * (x * y + z), 1.0 - 2.0 * (y**2 + z**2))
        return heading
   
    def get_angular_velocities(self, q: np.ndarray, dt: np.ndarray) -> np.ndarray:
        """
        Calculate the angular velocities from quaternions.
        
        Args:
            q (np.ndarray): Array of quaternions.
            dt (np.ndarray): Array of time differences between quaternions.

        Returns:
            np.ndarray: Angular velocities [roll_rate, pitch_rate, yaw_rate].
        """
        return (2 / dt) * np.array([
            q[:-1, 0] * q[1:, 1] - q[:-1, 1] * q[1:, 0] - q[:-1, 2] * q[1:, 3] + q[:-1, 3] * q[1:, 2],
            q[:-1, 0] * q[1:, 2] + q[:-1, 1] * q[1:, 3] - q[:-1, 2] * q[1:, 0] - q[:-1, 3] * q[1:, 1],
            q[:-1, 0] * q[1:, 3] - q[:-1, 1] * q[1:, 2] + q[:-1, 2] * q[1:, 1] - q[:-1, 3] * q[1:, 0]
        ])
    
    def derive_velocities(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Derive linear and angular velocities from pose and time buffers."""
        if len(self.pose_buffer) < 2:
            return np.zeros(3), np.zeros(3)

        # Calculate time differences
        dt = (self.time_buffer[-1] - self.time_buffer[0]) * 1e-9  # Nanoseconds to seconds
        if dt == 0:
            return np.zeros(3), np.zeros(3)

        # Calculate linear velocities
        linear_positions = np.array([
            [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]
            for pose in self.pose_buffer
        ])
        linear_velocities = np.diff(linear_positions, axis=0) / (dt / len(self.pose_buffer))
        avg_linear_velocity = np.mean(linear_velocities, axis=0)

        # Calculate angular velocities
        quaternions = np.array([
            [pose.pose.orientation.w, pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z]
            for pose in self.pose_buffer
        ])
        dt_per_step = np.ones((len(quaternions) - 1)) * (dt / (len(quaternions) - 1))
        angular_velocities = self.get_angular_velocities(quaternions, dt_per_step)
        avg_angular_velocity = np.mean(angular_velocities, axis=1)

        return avg_linear_velocity, avg_angular_velocity

    def build_state(self) -> torch.Tensor:
        """Build the state tensor for the task."""
        
        if self.robot_position is None or self.robot_quat is None:
            return None

        # Compute position error
        position_error = self.goal - self.robot_position[:2]
        position_dist = torch.norm(position_error)  # Euclidean distance to goal

        # Compute heading and target heading
        heading = self.quaternion_to_heading(self.robot_quat)
        target_heading = torch.atan2(position_error[1], position_error[0])
        target_heading_error = torch.atan2(torch.sin(target_heading - heading), torch.cos(target_heading - heading))

        lin_vel = torch.tensor(self.robot_vel[:2], dtype=torch.float32)
        yaw_rate = torch.tensor(self.robot_vel[5], dtype=torch.float32)
        self.state[0] = position_dist  # Scalar distance to goal
        self.state[1] = torch.cos(target_heading_error)
        self.state[2] = torch.sin(target_heading_error)
        self.state[3:5] = lin_vel  # Linear velocity
        self.state[5] = yaw_rate  # Yaw rate
        self.state[6:] = self.prev_action  # Previous action


    def model_inference(self, state: torch.Tensor) -> torch.Tensor:
        """Placeholder function for RL model inference."""
        # For demonstration purposes, return random actions.
        # Replace this with a trained model's inference.
        action = torch.rand(8, dtype=torch.float32) > 0.5
        # concat the air bearing value
        action = torch.cat([self.air_bearing, action], dim=0)

        
        return action

    def validate_task_completion(self) -> bool:
        """Validate if the task is successfully completed."""
        if self.robot_position is None:
            return False

        # Check if goal is reached within tolerance
        dist_to_goal = torch.norm(self.goal - self.robot_position[:2])
        if dist_to_goal < self.goal_tolerance:
            self.successful_steps += 1
        else:
            self.successful_steps = 0

        # Return True if goal is maintained for enough steps
        return self.successful_steps >= self.steps_to_validate

    def task_loop(self):
        """Main task loop."""
        self.build_state()
        if self.state is None:
            self.get_logger().warn("State could not be built. Waiting for pose data...")
            return

        # Perform inference to get the next action
        action = self.model_inference(self.state)
        self.get_logger().info(f"Step: {self.current_step}, State: {self.state.tolist()}, Action: {action.tolist()}")

        # Update the previous action
        self.prev_action = action

        # Check for task completion
        if self.validate_task_completion():
            self.get_logger().info("Task successfully completed!")
            self.timer.cancel()  # Stop the task loop

        self.current_step += 1


def main(args=None):
    rclpy.init(args=args)

    # Example task: "go_to_position" with goal (5.0, 5.0)
    task_node = RLTaskNode(task_name="go_to_position", goal=(5.0, 5.0))

    try:
        rclpy.spin(task_node)
    except KeyboardInterrupt:
        pass
    finally:
        task_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
