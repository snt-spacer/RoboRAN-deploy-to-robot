# Use ROS2 Humble base image
FROM ros:humble

# Set up working directory
WORKDIR /workspace

# Copy project files into the container
COPY . /workspace/

# Install Python dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    ros-humble-rclpy \
    ros-humble-geometry-msgs \
    byobu \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir -r /workspace/requirements.txt

# Source ROS2 setup
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Expose ports if needed
EXPOSE 11311

CMD ["bash"]
