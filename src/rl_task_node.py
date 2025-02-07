#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int16MultiArray, ByteMultiArray
import numpy as np
import torch
from typing import Tuple
import math
from skrl_inference.light_inference_runner import LightInferenceRunner
from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space, unflatten_tensorized_space
import yaml
import time
from gymnasium import spaces

class RLTaskNode(Node):
    def __init__(self, task_name: str, goal: Tuple[float, float]):
        super().__init__("rl_task_node")

        # Task details
        self.task_name = task_name
        self.goal = torch.tensor(goal, dtype=torch.float32)
        self.goal_tolerance = 0.1  # Distance tolerance for goal completion
        self.steps_to_validate = 10  # Steps to maintain the goal to validate success

        # State representation
        self.prev_action = torch.zeros(8, dtype=torch.float32)  # Shape (9,) [0: airbearing, 1-8: thrusters]
        self.state = torch.zeros(14, dtype=torch.float32)  # Shape (15,) [0: target_dist, 1-2: target_heading, 3-4: lin_vel, 5: yaw_rate, 6-15: prev_action]
        self.air_bearing = torch.tensor((1,), dtype=torch.float32)  # Placeholder for air bearing value    

        # Robot state variables
        self.robot_position = None
        self.robot_quat = None
        self.robot_vel = np.zeros(6)  # Linear (x, y, z) + Angular (roll, pitch, yaw)

        # Task execution tracking
        self.current_step = 0
        self.successful_steps = 0
        self.num_steps_episode = 200

        # Buffers for velocity computation
        self.pose_buffer = []
        self.time_buffer = []

        # ROS2 communication (/omniFPS/Robots/FloatingPlatform/PoseStamped)
        # self.create_subscription(PoseStamped, "/vrpn_client_node/FP_exp_RL/pose", self.pose_callback, 10)
        self.create_subscription(PoseStamped, "/omniFPS/Robots/FloatingPlatform/PoseStamped", self.pose_callback, 1)
        self.create_subscription(PoseStamped, "/FloatingPlatform/goal", self.goal_callback, 1)
        # ROS2 communication (MultiBinaryArray message for thruster commands)
        self.action_pub = self.create_publisher(ByteMultiArray, "/omniFPS/Robots/FloatingPlatform/thrusters/input", 1)
        # self.action_pub = self.create_publisher(Int16MultiArray, "/omniFPS/Robots/FloatingPlatform/thrusters/input", 1)

        self.thruster_msg = ByteMultiArray()
        # self.thruster_msg = Int16MultiArray()
        # Timer for repeating task loop (10 Hz)
        self.timer = self.create_timer(0.1, self.task_loop) # TODO: Change to frequency variable (to be called from a launch file)

        # Load the RL model
        model_path = "models/FP_GoToPosition"   
        self.load_model(model_path)

        self.get_logger().info(f"RL Task Node initialized for task: {self.task_name}")
        
    def goal_callback(self, msg: PoseStamped):
        """Callback for goal position messages."""
        self.goal = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]) # Update goal position
        self.get_logger().info(f"Goal position updated: {self.goal.tolist()}")

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
        heading = math.atan2(2.0 * (x * y + z), 1.0 - 2.0 * (y*y + z*z))
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

    def get_rotation_matrix(self, heading: float) -> np.ndarray:
        """Get the 2D rotation matrix for a given heading angle."""
        return np.array([
            [np.cos(heading), -np.sin(heading)],
            [np.sin(heading), np.cos(heading)]
        ])

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

        R = self.get_rotation_matrix(-heading) # Rotate to robot frame 

        lin_vel = torch.tensor(R@self.robot_vel[:2], dtype=torch.float32)
        yaw_rate = torch.tensor(self.robot_vel[5], dtype=torch.float32)
        self.state[0] = position_dist  # Scalar distance to goal
        self.state[1] = torch.cos(target_heading_error)
        self.state[2] = torch.sin(target_heading_error)
        self.state[3:5] = lin_vel  # Linear velocity
        self.state[5] = yaw_rate  # Yaw rate
        self.state[6:] = self.prev_action  # Previous action

    def load_model(self, log_dir: str):
        """Load the RL model from a given path."""
        # Placeholder function for loading the RL model

        env_params_path = f"{log_dir}/params/env.yaml"
        with open(env_params_path) as f:
            env_cfg = yaml.load(f, Loader=yaml.FullLoader)

        agent_params_path = f"{log_dir}/params/agent.yaml"
        with open(agent_params_path) as f:
            agent_params = yaml.safe_load(f)
        print(env_cfg)

        self.player = LightInferenceRunner(env_cfg, agent_params)
        resume_path = log_dir + "/checkpoints/best_agent.pt"

        self.player.build(resume_path)
        self.action_space = spaces.Tuple([spaces.Discrete(2)] * 8)
        
    def model_inference(self, state: torch.Tensor) -> torch.Tensor:
        """Placeholder function for RL model inference."""
        # For demonstration purposes, return random or constant actions.
        #action = torch.rand(8, dtype=torch.float32) > 0.5 # random actions
        # action = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.float32) # constant actions
        # concat the air bearing value
        """Performs model inference and returns binary thruster actions."""
        with torch.inference_mode():
            actions = self.player.act(state, timestep=0, timesteps=0)[0]
        # Ensure the output is correctly shaped
        actions = unflatten_tensorized_space(self.action_space, actions)
        # Convert to tensor and enforce binary values
        actions = torch.cat(actions).squeeze()
        #print min and max values of the actions
        print(f"Min value of actions: {actions.min()}")
        print(f"Max value of actions: {actions.max()}")
        # Apply binary thresholding: Any value > 0.5 becomes 1, else 0
        binary_actions = (actions > 0.5).int()

        # Concatenate the air bearing value
        action = torch.cat([self.air_bearing, binary_actions], dim=0)

        print(f"Final Binary Actions: {action.tolist()}")
        return action


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
  
    def turn_off_thrusters(self):
        """Turn off all thrusters."""
        # turn-off all trhusters and 1 sec delay later the air-bearing
        self.thruster_msg.data = [1, 0, 0, 0, 0, 0, 0, 0, 0]
        self.action_pub.publish(self.thruster_msg)
        self.get_logger().info("Thrusters turned off.")
        self.thruster_msg.data = [0] * 9 
        # wait 1 sec and then turn off the air-bearing (use the ros node to sleep)
        time.sleep(1)
        self.action_pub.publish(self.thruster_msg)
        self.get_logger().info("Air-bearing turned off.")
        
    
    def task_loop(self):
        """Main task loop."""
        self.build_state()
        if self.state is None:
            self.get_logger().warn("State could not be built. Waiting for pose data...")
            return

        # Perform inference to get the next action
        action = self.model_inference(self.state)
        # Publish the action to the robot (e.g., thruster commands using ByteMultiArray -- need to convert to bytes)
        # byte_action = [bytes([value]) for value in action.int().tolist()]
        self.thruster_msg.data = [value.to_bytes(1, byteorder='little') for value in action.int().tolist()]
        # Publish the action as Int16MultiArray
        # self.thruster_msg.data = action.int().tolist()

        self.action_pub.publish(self.thruster_msg)
        # Log the step details
        self.get_logger().info(f"Step: {self.current_step}, State: {self.state.tolist()}, Action: {action.tolist()}")

        # Update the previous action (removing the air bearing value)

        self.prev_action = action[1:]

        self.current_step += 1

        # create condition to stop the loop (after num_steps_episode are reached or when task completion)
        if self.validate_task_completion():
            self.get_logger().info("Task successfully completed!")
            self.timer.cancel()  # Stop the task loop
            # self.turn_off_thrusters()


        if self.current_step >= self.num_steps_episode:
            self.get_logger().info("Task timed out!")
            self.timer.cancel()
            # self.turn_off_thrusters()


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
