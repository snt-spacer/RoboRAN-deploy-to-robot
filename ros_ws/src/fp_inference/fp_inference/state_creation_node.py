#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Float32MultiArray
import numpy as np
from typing import List, Tuple


class StateCreationNode(Node):
    def __init__(self):
        super().__init__("state_creation_node")

        # Subscriptions
        self.pose_sub = self.create_subscription(
            PoseStamped, "/vrpn_client_node/FP_exp_RL/pose", self.pose_callback, 10
        )
        self.goal_sub = self.create_subscription(Point, "/spacer_floating_platform/goal", self.goal_callback, 10)

        # Publisher
        self.state_pub = self.create_publisher(Float32MultiArray, "/rl_task_state", 10)

        # Timer (10 Hz publishing)
        self.timer = self.create_timer(0.1, self.publish_state)

        # Internal state variables
        self.robot_position = None
        self.robot_quat = None
        self.robot_vel = np.zeros(6)  # Linear (x, y, z) and Angular (roll, pitch, yaw)
        self.goal_position = None
        self.pose_buffer = []
        self.time_buffer = []

    def pose_callback(self, msg: PoseStamped):
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

    def goal_callback(self, msg: Point):
        # Update goal position
        self.goal_position = np.array([msg.x, msg.y, msg.z])

    def quaternion_to_heading(self, quat):
        """
        Converts a quaternion to a heading (yaw angle in radians).
        """
        _, x, y, z = quat
        heading = np.arctan2(2.0 * (x * y + z), 1.0 - 2.0 * (y**2 + z**2))
        return heading

    def angular_velocities(self, q: np.ndarray, dt: np.ndarray) -> np.ndarray:
        """
        Calculate the angular velocities from quaternions.

        Args:
            q (np.ndarray): Array of quaternions.
            dt (np.ndarray): Array of time differences between quaternions.

        Returns:
            np.ndarray: Angular velocities [roll_rate, pitch_rate, yaw_rate].
        """
        return (2 / dt) * np.array(
            [
                q[:-1, 0] * q[1:, 1] - q[:-1, 1] * q[1:, 0] - q[:-1, 2] * q[1:, 3] + q[:-1, 3] * q[1:, 2],
                q[:-1, 0] * q[1:, 2] + q[:-1, 1] * q[1:, 3] - q[:-1, 2] * q[1:, 0] - q[:-1, 3] * q[1:, 1],
                q[:-1, 0] * q[1:, 3] - q[:-1, 1] * q[1:, 2] + q[:-1, 2] * q[1:, 1] - q[:-1, 3] * q[1:, 0],
            ]
        )

    def derive_velocities(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Derive the linear and angular velocities using pose and time buffers.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Linear and angular velocities.
        """
        if len(self.pose_buffer) < 2:
            return np.zeros(3), np.zeros(3)

        # Calculate time differences
        dt = (self.time_buffer[-1] - self.time_buffer[0]) * 1e-9  # Nanoseconds to seconds
        if dt == 0:
            return np.zeros(3), np.zeros(3)

        # Calculate linear velocities
        linear_positions = np.array(
            [[pose.pose.position.x, pose.pose.position.y, pose.pose.position.z] for pose in self.pose_buffer]
        )
        linear_velocities = np.diff(linear_positions, axis=0) / (dt / len(self.pose_buffer))
        avg_linear_velocity = np.mean(linear_velocities, axis=0)

        # Calculate angular velocities
        quaternions = np.array(
            [
                [pose.pose.orientation.w, pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z]
                for pose in self.pose_buffer
            ]
        )
        dt_per_step = np.ones((len(quaternions) - 1)) * (dt / (len(quaternions) - 1))
        angular_velocities = self.angular_velocities(quaternions, dt_per_step)
        avg_angular_velocity = np.mean(angular_velocities, axis=1)

        return avg_linear_velocity, avg_angular_velocity

    def compute_state(self):
        if self.robot_position is None or self.goal_position is None:
            return None

        # Compute state for `go_to_position` task
        position_error = self.goal_position[:2] - self.robot_position[:2]
        position_dist = np.linalg.norm(position_error)

        # Compute heading and target heading
        heading = self.quaternion_to_heading(self.robot_quat)
        target_heading = np.arctan2(position_error[1], position_error[0])
        target_heading_error = np.arctan2(np.sin(target_heading - heading), np.cos(target_heading - heading))

        state = np.zeros(7)
        state[0] = position_dist
        state[1] = np.cos(target_heading_error)
        state[2] = np.sin(target_heading_error)
        state[3:5] = self.robot_vel[:2]  # Linear velocity
        state[5] = self.robot_vel[5]  # Yaw angular velocity
        state[6] = 0.0  # Placeholder for previous action

        return state

    def publish_state(self):
        state = self.compute_state()
        if state is not None:
            # Publish state
            state_msg = Float32MultiArray(data=state.tolist())
            self.state_pub.publish(state_msg)
            self.get_logger().info(f"Publishing state: {state}")


def main(args=None):
    rclpy.init(args=args)
    node = StateCreationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
