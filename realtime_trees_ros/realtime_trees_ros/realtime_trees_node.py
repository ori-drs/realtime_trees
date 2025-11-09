#!/usr/bin/python3

import rclpy
from realtime_trees_ros.realtime_trees import RealtimeTrees


def main(args=None):
    rclpy.init(args=args)

    realtime_trees = RealtimeTrees()
    realtime_trees.get_logger().info("Spinning...")

    try:
        rclpy.spin(realtime_trees)
    except KeyboardInterrupt:
        print("Node has been interrupted, shutting down...")
    finally:
        # Perform cleanup here if needed
        realtime_trees.on_shutdown()
        realtime_trees.destroy_node()
        # See https://github.com/ros2/rclpy/issues/1081
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()