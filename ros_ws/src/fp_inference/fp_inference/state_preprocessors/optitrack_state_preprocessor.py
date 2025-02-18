from geometry_msgs.msg import PoseStamped
import torch
import copy

from .base_state_preprocessor import BaseStatePreProcessor
from . import Registerable

import rclpy


class OptitrackStatePreProcessor(Registerable, BaseStatePreProcessor):
    """A class to process state information from a robot. The state processor maintains a buffer of the robot's position,
    quaternion, and time. The state processor can compute the heading, rotation matrix, linear velocities in the world frame,
    and linear velocities in the body frame. The state processor is primed after a certain number of steps, after which the
    state can be accessed through the state variables.
    All the operations are done on the GPU to ensure mimum CPU-GPU data transfer.
    """

    def __init__(self, buffer_size: int = 30, device: str = "cuda", **kwargs):
        """Initialize the state processor with a buffer size and device.

        Args:
            buffer_size (int, optional): The size of the state buffer. Defaults to 30.
            device (str, optional): The device to perform computations on. Defaults to "cuda".
        """
        super().__init__(device=device, buffer_size=buffer_size, **kwargs)

        # General parameters
        self.ROS_TYPE = PoseStamped

        # State buffers for position, quaternion, and time
        self._position_buffer: torch.Tensor = torch.zeros((buffer_size, 3), device=device)
        self._quaternion_buffer: torch.Tensor = torch.zeros((buffer_size, 4), device=device)
        self._time_buffer: torch.Tensor = torch.zeros((buffer_size), device=device, dtype=torch.float64)

    def update_angular_velocities(self):
        """Compute angular velocities from quaternions."""

        # Take time buffer extremas and convert to seconds
        dt = (self._time_buffer[-1] - self._time_buffer[0]) / self._buffer_size
        # Compute quaternion differences and divide by time to get angular velocities
        vel = (2 / dt) * torch.stack(
            [
                self._quaternion_buffer[:-1, 0] * self._quaternion_buffer[1:, 1]
                - self._quaternion_buffer[:-1, 1] * self._quaternion_buffer[1:, 0]
                - self._quaternion_buffer[:-1, 2] * self._quaternion_buffer[1:, 3]
                + self._quaternion_buffer[:-1, 3] * self._quaternion_buffer[1:, 2],
                self._quaternion_buffer[:-1, 0] * self._quaternion_buffer[1:, 2]
                + self._quaternion_buffer[:-1, 1] * self._quaternion_buffer[1:, 3]
                - self._quaternion_buffer[:-1, 2] * self._quaternion_buffer[1:, 0]
                - self._quaternion_buffer[:-1, 3] * self._quaternion_buffer[1:, 1],
                self._quaternion_buffer[:-1, 0] * self._quaternion_buffer[1:, 3]
                - self._quaternion_buffer[:-1, 1] * self._quaternion_buffer[1:, 2]
                + self._quaternion_buffer[:-1, 2] * self._quaternion_buffer[1:, 1]
                - self._quaternion_buffer[:-1, 3] * self._quaternion_buffer[1:, 0],
            ]
        )
        # Update angular velocities in the world frame
        self._angular_velocities_world = torch.mean(vel, dim=1).unsqueeze(0)

    def update_linear_velocities(self):
        # Take time buffer extremas and convert to seconds
        dt = (self._time_buffer[-1] - self._time_buffer[0]) / self._buffer_size
        # Compute position differences and divide by time to get velocities
        vel = torch.diff(self._position_buffer, dim=0) / (dt)
        # Update linear velocities in the world frame
        self._linear_velocities_world = torch.mean(vel, dim=0).unsqueeze(0)

    @staticmethod
    def append_left_tensor_queue(queue_tensor: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        """Append a tensor to the left of a queue tensor.

        Args:
            queue_tensor (torch.Tensor): The queue tensor to append the tensor to.
            tensor (torch.Tensor): The tensor to append to the queue tensor.

        Returns:
            torch.Tensor: The updated queue tensor."""

        queue_tensor = torch.cat((tensor.unsqueeze(0), queue_tensor[:-1]))
        return queue_tensor

    @staticmethod
    def append_right_tensor_queue(queue_tensor: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        """Append a tensor to the right of a queue tensor.

        Args:
            queue_tensor (torch.Tensor): The queue tensor to append the tensor to.
            tensor (torch.Tensor): The tensor to append to the queue tensor.

        Returns:
            torch.Tensor: The updated queue tensor."""

        queue_tensor = torch.cat((queue_tensor[1:], tensor.unsqueeze(0)))
        return queue_tensor

    def update_state_ROS(self, pose: PoseStamped, **kwargs) -> None:
        """Update the state processor with a new pose message. If the state processor is primed, update the state.
        When primed, the state can be accessed through the state variables.

        Args:
            pose (PoseStamped): The pose message to update the state processor with."""

        # Convert ROS message to tensors
        position = torch.tensor(
            [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z], device=self._device
        )
        quaternion = torch.tensor(
            [pose.pose.orientation.w, pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z],
            device=self._device,
        )
        self._last_time = copy.copy(self._time)
        if pose.header.stamp.sec == 0:
            clock = rclpy.clock.Clock().now().to_msg()
            self._time = clock.sec + clock.nanosec * 1e-9
        else:
            self._time = pose.header.stamp.sec + pose.header.stamp.nanosec * 1e-9    
        # Update buffers
        self._position_buffer = self.append_right_tensor_queue(self._position_buffer, position)
        self._quaternion_buffer = self.append_right_tensor_queue(self._quaternion_buffer, quaternion)
        self._time_buffer = self.append_right_tensor_queue(
            self._time_buffer, torch.tensor(copy.copy(self._time), device=self._device, dtype=torch.float64)
        )
        # Update the step count and priming status
        self._step += 1
        self._is_primed = self._step >= self._buffer_size

        # If the state processor is primed, update the state
        if self._is_primed:
            # Update the position and quaternion
            self._position = position.unsqueeze(0)
            self._quaternion = quaternion.unsqueeze(0)
            # Update velocities
            self.update_linear_velocities()
            self.update_angular_velocities()

    def reset(self) -> None:
        """Reset the state processor to its initial state."""

        super().reset()

        # Reset state buffers
        self._position_buffer.fill_(0)
        self._quaternion_buffer.fill_(0)
        self._time_buffer.fill_(0)
