# LangGraph PyBullet Simulation

This repository demonstrates how to combine [LangGraph](https://github.com/langchain-ai/langgraph) with
[LangChain](https://github.com/langchain-ai/langchain) to control a PyBullet simulation. A simple
LangGraph agent plans and executes goals such as moving a cube-shaped robot toward colored balls.

In addition to the PyBullet demo, the repository now also contains a ready-to-use ROS 2 / Gazebo
simulation of a differential drive robot equipped with both a standard RGB camera and a panoramic
360° camera that can be driven with `geometry_msgs/Twist` velocity commands.

## Features

- PyBullet environment containing a plane, a cube robot, and two spheres (red and blue).
- LangGraph planning loop powered by an Azure OpenAI chat model.
- Tool-enabled agent that can move, turn, and evaluate the simulated scene through registered
  LangChain tools.
- ROS 2 package (`dual_camera_bot`) that launches Gazebo with an RGB + 360° camera robot exposing a
  `cmd_vel` interface for manual or programmatic control.

## Setup

### PyBullet + LangGraph demo

Create a virtual environment and install the latest LangGraph, LangChain, and PyBullet releases:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install langgraph langchain pybullet numpy python-dotenv pillow
```

### ROS 2 Gazebo simulation

The Gazebo portion assumes you have a ROS 2 distribution (Humble or newer) with
`gazebo_ros`, `xacro`, and `ros-<distro>-gazebo-ros-pkgs` installed. Create (or reuse) a colcon
workspace and clone this repository into its `src` directory:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/your-user/langgrapgh_sim.git
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select dual_camera_bot
source install/setup.bash
```

## Running the simulations

### PyBullet + LangGraph

Before running the simulation, export the Azure OpenAI environment variables so the agent can reach
your deployment:

```bash
export OPENAI_API_KEY="<your-azure-openai-key>"
export OPENAI_API_VERSION="<api-version>"
export AZURE_OPENAI_ENDPOINT="https://<your-resource-name>.openai.azure.com"
export OPENAI_MODEL_NAME="<deployment-name>"
```

Then start the simulator:

```bash
python main.py        # add --gui if you want the PyBullet GUI
```

Pass `--action-delay <seconds>` to pause briefly after each agent tool execution.

Pass `--gui` to launch the PyBullet GUI (if your environment supports it).

Once the program starts you can type instructions such as:

```bash
Goal> approach the red ball
```

The LangGraph agent will iteratively plan and execute until it reaches the requested target. Type
`quit` (or press `Ctrl+C`) to exit.

### Gazebo dual-camera robot

Launch Gazebo with the dual camera robot:

```bash
ros2 launch dual_camera_bot sim.launch.py
```

This spawns a differential drive robot in an empty world and publishes:

- `/dual_camera_bot/rgb/image_raw` and `/dual_camera_bot/rgb/camera_info` for the forward-facing RGB
  camera
- `/dual_camera_bot/panorama/image_raw` and `/dual_camera_bot/panorama/camera_info` for the 360°
  fisheye camera
- `/dual_camera_bot/cmd_vel` for velocity commands consumed by the Gazebo diff-drive plugin

To drive the robot from the command line, you can use the standard teleoperation package:

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args --remap cmd_vel:=/dual_camera_bot/cmd_vel
```

Any node that publishes `geometry_msgs/Twist` messages to `/dual_camera_bot/cmd_vel` can control the
robot in simulation. The diff-drive plugin also publishes odometry to `/dual_camera_bot/odom`.
