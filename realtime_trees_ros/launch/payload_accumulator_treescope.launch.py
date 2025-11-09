import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, EnvironmentVariable
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile
from ament_index_python.packages import get_package_share_directory, get_package_prefix


def generate_launch_description() -> LaunchDescription:
    parameters_file = LaunchConfiguration('params')
    use_sim_time = LaunchConfiguration('use_sim_time')

    default_use_sim_time = EnvironmentVariable('USE_SIM_TIME', default_value='false')
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value=default_use_sim_time,
        description='Use simulation time'
    )
    default_parameters_file = os.path.join(
        get_package_share_directory('realtime_trees_ros'),
        'config',
        'payload_accumulator_config_treescope.yaml'
    )
    payload_accumulator_parameters_arg = DeclareLaunchArgument(
        'params',
        default_value=default_parameters_file,
        description='Parameters file to be used'
    )
    
    payload_accumulator_node = Node(
        package='realtime_trees_ros',
        executable='payload_accumulator_node',
        parameters=[ParameterFile(parameters_file, allow_substs=True),
                    {'use_sim_time': use_sim_time}],
        output='screen',
        emulate_tty=True
    )

    ld = LaunchDescription()
    ld.add_action(use_sim_time_arg)
    ld.add_action(payload_accumulator_parameters_arg)
    ld.add_action(payload_accumulator_node)
    return ld