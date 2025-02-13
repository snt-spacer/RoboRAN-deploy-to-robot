# Use ROS2 Humble base image
FROM ros:humble

# Set the working directory to the main folder of your repository
WORKDIR /RANS_DeployToRobot

# Copy everything from the repository into the working directory
COPY . .

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    ros-humble-rclpy \
    ros-humble-geometry-msgs \
    byobu \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

RUN git clone https://github.com/elharirymatteo/skrl.git skrl
RUN pip3 install -e skrl/.

# Source ROS2 setup
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

WORKDIR /RANS_DeployToRobot/ros2_ws
RUN colcon build

# Source ROS2 workspace setup
RUN echo "source /RANS_DeployToRobot/ros2_ws/install/setup.bash" >> ~/.bashrc

# Default command
CMD ["bash"]
