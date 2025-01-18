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
└── .gitignore                      # Ignored files (e.g., __pycache__, logs)
