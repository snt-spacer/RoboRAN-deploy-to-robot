from nav_msgs.msg import Odometry
import torch
import copy

from .base_state_preprocessor import BaseStatePreProcessor
from . import Registerable

class OdometryStatePreProcessor(Registerable, BaseStatePreProcessor):
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
        self.ROS_TYPE = Odometry

        # Always primed, no waiting for buffer
        self._is_primed = True

    @property
    def linear_velocities_world(self) -> torch.Tensor | None:
        """Return the linear velocities in the world frame. If the velocities are not available, return None.
        
        Returns:
            torch.Tensor[N, 3] | None: The linear velocities in the world frame if available, otherwise None."""
        
        if (self.linear_velocities_world is not None) and (self._step_linear_velocity_body != self._step):
            self.get_linear_velocities_world()
            # Update the step count for lazy updates
            self._step_linear_velocity_body = copy.copy(self._step)
        return self._linear_velocities_world

    @property
    def angular_velocities_world(self) -> torch.Tensor | None:
        """Return the angular velocities in the world frame. If the velocities are not available, return None.
        
        Returns:
            torch.Tensor[N, 3] | None: The angular velocities in the world frame if available, otherwise None."""
        
        if (self.angular_velocities_world is not None) and (self._step_angular_velocity_body != self._step):
            self._angular_velocities_world = self._angular_velocities_body
            # Update the step count for lazy updates
            self._step_angular_velocity_body = copy.copy(self._step)
        return self._angular_velocities_world

    @property
    def linear_velocities_body(self) -> torch.Tensor | None:
        """Return the linear velocities in the body frame. If the velocities are not available, return None.
        
        Returns:
            torch.Tensor[N, 3] | None: The linear velocities in the body frame if available, otherwise None."""
        
        return self._linear_velocities_body

    @property
    def angular_velocities_body(self) -> torch.Tensor | None:
        """Return the angular velocities in the body frame. If the velocities are not available, return None.
        Note: The angular velocities in the body frame are the same as the angular velocities in the world frame.
        
        Returns:
            torch.Tensor[N, 3] | None: The angular velocities in the body frame if available, otherwise None."""
        
        return self._angular_velocities_body
    
    def get_linear_velocities_world(self) -> torch.Tensor:
        """Compute the linear velocities in the world frame from the linear velocities in the body frame."""

        # Get heading & rotation matrix
        inv_rotation_matrix = torch.linalg.pinv(self.rotation_matrix)
        # Cast the velocities to the world frame
        self._linear_velocities_world[:,:2] = inv_rotation_matrix @ self._linear_velocities_body.unsqueeze(1)

    def update_state_ROS(self, odometry: Odometry, **kwargs) -> None:
        """Update the state processor with a new odometry message. If the state processor is primed, update the state.
        When primed, the state can be accessed through the state variables.

        The odometry message provides the pose in the world frame, and the velocities in the body frame.
        
        Args:
            odometry (Odometry): The odometry message to update the state processor with."""

        # Convert ROS message to tensors
        self._position[0] = torch.tensor([odometry.pose.pose.position.x, odometry.pose.pose.position.y, odometry.pose.pose.position.z], device=self._device)
        self._quaternion[0] = torch.tensor([odometry.pose.pose.orientation.w, odometry.pose.pose.orientation.x, odometry.pose.pose.orientation.y, odometry.pose.pose.orientation.z], device=self._device)
        self._linear_velocities_body = torch.tensor([odometry.twist.twist.linear.x, odometry.twist.twist.linear.y, odometry.twist.twist.linear.z], device=self._device)
        self._angular_velocities_body = torch.tensor([odometry.twist.twist.angular.x, odometry.twist.twist.angular.y, odometry.twist.twist.angular.z], device=self._device)
        # Update the step count and priming status
        self._step += 1
        self._time = odometry.header.stamp.sec + odometry.header.stamp.nanosec * 1e-9


    def reset(self):
        super().reset()

        self._is_primed = True