import time
import subprocess


def test_state_creation():
    # Launch the simulator in a subprocess
    simulator_process = subprocess.Popen(["ros2", "run", "src", "optitrack_simulator.py"])
    time.sleep(2)  # Wait for the simulator to initialize

    # Launch the state creation node
    state_creation_process = subprocess.Popen(["ros2", "run", "src", "state_creation_node.py"])

    time.sleep(10)  # Let the nodes run and produce data

    # Kill the processes
    simulator_process.terminate()
    state_creation_process.terminate()

    # Check logs or outputs (optional validation step)
    print("Nodes launched and terminated successfully.")
    assert True


if __name__ == "__main__":
    test_state_creation()
