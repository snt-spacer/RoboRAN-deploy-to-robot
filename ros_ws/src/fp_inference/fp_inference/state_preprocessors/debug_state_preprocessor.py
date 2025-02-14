from std_msgs.msg import Empty
from . import Registerable
from . import BaseStatePreProcessor
import torch
import copy


class DebugStatePreProcessor(Registerable, BaseStatePreProcessor):
    """A class to process state information from a robot. The state processor maintains a buffer of the robot's position,
    quaternion, and time. The state processor can compute the heading, rotation matrix, linear velocities in the world frame,
    and linear velocities in the body frame. The state processor is primed after a certain number of steps, after which the
    state can be accessed through the state variables.
    All the operations are done on the GPU to ensure mimum CPU-GPU data transfer.
    """

    def __init__(self, buffer_size: int = 30, device: str = "auto", **kwargs):
        """Initialize the state processor with a buffer size and device.

        Args:
            buffer_size (int, optional): The size of the state buffer. Defaults to 30.
            device (str, optional): The device to perform computations on. Defaults to "auto".
        """

        self.ROS_TYPE = Empty
        self.ROS_CALLBACK = self.update_state_ROS
        self.ROS_QUEUE_SIZE = 1

        super().__init__(buffer_size=buffer_size, device=device, **kwargs)

    def update_state_ROS(self, *args, **kwargs) -> None:
        """Update the state processor with a new ROS message. If the state processor is primed, update the state.
        When primed, the state can be accessed through the state variables."""

        self._position = torch.zeros((1, 3), device=self._device)
        self._quaternion = torch.zeros((1, 4), device=self._device)
        self._time = torch.zeros((1), device=self._device)
        self._linear_velocities_world = torch.zeros((1, 3), device=self._device)
        self._angular_velocities_world = torch.zeros((1, 3), device=self._device)
        self._step += 1

