from rclpy.node import Node
from nav_msgs.msg import Path
from vilens_msgs.msg import PoseGraph


class VilensPosegraphRepublisher(Node):
    def __init__(self):
        super().__init__("vilens_posegraph_republisher")

        # Subscribers
        self._sub_posegraph = self.create_subscription(
            PoseGraph, "/vilens_slam/pose_graph", self.posegraph_callback, 10
        )

        # Publishers
        self._pub_posegraph_update = self.create_publisher(
            Path, "/pose_graph", 10
        )

    def posegraph_callback(self, msg: PoseGraph):
        self._pub_posegraph_update.publish(msg.path)