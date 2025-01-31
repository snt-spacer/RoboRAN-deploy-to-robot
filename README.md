# RL Task State Creation and Inference Framework

This repository extends the [IsaacLab RANS project](https://github.com/SpaceR-x-DreamLab-RL/IsaacLab_RANS) to provide a modular and scalable framework for creating RL task states, running inference, and simulating robot control tasks. The system supports seamless integration with ROS, Docker, and a simulated OptiTrack environment for rapid testing and deployment.

## Features

- **State Creation Node**: Computes the RL policy input (state vector) for a selected task and robot.
- **Model Inference Node**: Subscribes to the state topic, runs the RL policy, and publishes actions to control the robot.
- **Simulated OptiTrack Data**: Generates synthetic pose updates to test the framework without real hardware.
- **Dockerized Setup**: Ensures consistent environments for development, testing, and deployment.

---

## Repository Structure

```plaintext
RANS_DeplyToRobot/
├── docker/
│   ├── Dockerfile                  # Main environment for ROS + model
│   └── entrypoint.sh               # Entrypoint script for container
│  
├── src/
│   ├── state_creation_node.py      # ROS node for state creation
│   ├── optitrack_simulator.py      # Simulated OptiTrack messages
│   ├── model_inference_node.py     # ROS node for running RL models
│   └── utils.py                    # Shared utility functions
├── models/
│   ├── policy_model.pth            # Pretrained RL model weights
│   └── model_definition.py         # RL model architecture definition
├── config/
│   ├── ros_config.yaml             # ROS topic and parameters
│   ├── task_config.yaml            # Task-specific configurations
│   └── docker_config.yaml          # Docker-specific configurations
├── tests/
│   ├── test_state_creation.py      # Unit tests for state creation node
│   ├── test_model_inference.py     # Unit tests for inference node
│   └── test_utils.py               # Unit tests for shared utilities
├── requirements.txt                # Python dependencies
├── README.md                       
└── .gitignore                      # Ignored files 
```


## **Docker Installation and Usage Guide**

This module is containerized using Docker to simplify installation and runtime dependencies. Follow the steps below to set up and run the module inside a Docker container.

---

### **Prerequisites**
- Install [Docker](https://docs.docker.com/get-docker/):
  - For Ubuntu:
    ```bash
    sudo apt update
    sudo apt install docker.io
    sudo systemctl start docker
    sudo systemctl enable docker
    ```
  - For macOS or Windows: Follow the instructions on the [Docker website](https://docs.docker.com/get-docker/).

- (Optional) Install [Docker Compose](https://docs.docker.com/compose/install/) if you plan to use multiple services.

---

### **Building the Docker Image**

1. **Navigate to the Repository Folder**:
   ```bash
   cd ~/Projects/RANS_DeployToRobot
   ```

2. **Build the Docker Image**:
   Run the following command to build the image:
   ```bash
   docker build -t rans_deploytorobot .
   ```

   - `-t rans_deploytorobot`: Tags the image with the name `rans_deploytorobot`.
   - `.`: Refers to the current directory, which contains the `Dockerfile`.

---

### **Running the Docker Container**

1. **Launch the Container**:
   Run the container interactively:
   ```bash
   docker run -it --rm --name rans_test rans_deploytorobot
   ```

   - `-it`: Starts the container interactively.
   - `--rm`: Removes the container after you exit.
   - `--name rans_test`: Assigns the container the name `rans_test`.

2. **Verify Files Inside the Container**:
   Once inside the container, ensure the repository files are available:
   ```bash
   ls
   ```
   You should see:
   ```
   config  docker  models  README.md  src  tests  utils
   ```

3. **Run the ROS2 Nodes**:
   - Start the **OptiTrack Simulator**:
     ```bash
     python3 src/optitrack_simulator.py
     ```
   - Start the **Goal Publisher**:
     ```bash
     python3 src/goal_publisher.py
     ```
   - Start the **State Creation Node**:
     ```bash
     python3 src/state_creation_node.py
     ```

---

### **Viewing Published Topics**

1. List all active topics:
   ```bash
   ros2 topic list
   ```

2. Echo messages from a specific topic:
   - For the goal:
     ```bash
     ros2 topic echo /spacer_floating_platform/goal
     ```
   - For the state:
     ```bash
     ros2 topic echo /rl_task_state
     ```

---

### **Stopping the Container**

- If the container is running interactively, exit by typing:
  ```bash
  exit
  ```

- If you detached from the container, stop it with:
  ```bash
  docker stop rans_test
  ```

---

### **Optional: Mount Local Directory**

If you want to test code changes without rebuilding the Docker image every time, you can mount your local repository into the container:

```bash
docker run -it --rm --name rans_test -v $(pwd):/RANS_DeployToRobot rans_deploytorobot
```

This will:
- Mount your local directory to `/RANS_DeployToRobot` inside the container.
- Reflect any changes made locally in the container.

---

### **Cleaning Up Docker Resources**

To free up space on your system:

- Remove unused Docker images:
  ```bash
  docker image prune
  ```

- List all Docker containers:
  ```bash
  docker ps -a
  ```

- Remove stopped containers:
  ```bash
  docker container prune
  ```

---

### **Common Issues**

1. **Cannot Find File in Container**:
   - Ensure the `COPY . .` line in the `Dockerfile` properly copies all files into the container.

2. **ROS2 Topics Not Found**:
   - Verify all nodes are running:
     ```bash
     docker exec -it rans_test bash
     ros2 topic list
     ```

