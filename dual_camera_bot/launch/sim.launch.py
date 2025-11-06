from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("dual_camera_bot")

    declared_arguments = [
        DeclareLaunchArgument(
            "world",
            default_value=PathJoinSubstitution([package_share, "worlds", "dual_camera.world"]),
            description="SDF world file to load in Gazebo",
        ),
        DeclareLaunchArgument(
            "model",
            default_value=PathJoinSubstitution([package_share, "description", "dual_camera_bot.urdf.xacro"]),
            description="Path to the dual camera robot xacro file",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation clock",
        ),
        DeclareLaunchArgument(
            "robot_name",
            default_value="dual_camera_bot",
            description="Name of the spawned Gazebo entity",
        ),
    ]

    world = LaunchConfiguration("world")
    model = LaunchConfiguration("model")
    use_sim_time = LaunchConfiguration("use_sim_time")
    robot_name = LaunchConfiguration("robot_name")

    robot_description = Command(["xacro ", model])

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("gazebo_ros"),
                "launch",
                "gazebo.launch.py",
            ])
        ),
        launch_arguments={"world": world}.items(),
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time, "robot_description": robot_description}],
    )

    spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=["-entity", robot_name, "-topic", "robot_description"],
        output="screen",
    )

    return LaunchDescription(declared_arguments + [gazebo, robot_state_publisher, spawn_robot])
