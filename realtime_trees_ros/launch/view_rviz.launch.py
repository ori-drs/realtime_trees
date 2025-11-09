import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    package_dir = get_package_share_directory('realtime_trees_ros')
    rviz_config_file = os.path.join(package_dir, 'rviz', 'realtime_trees.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='realtime_trees_rviz',
        arguments=['-d', rviz_config_file],
        output='screen'
    )

    ld = LaunchDescription()
    ld.add_action(rviz_node)
    return ld