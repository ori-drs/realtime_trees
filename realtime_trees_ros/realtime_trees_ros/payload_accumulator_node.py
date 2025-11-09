import rclpy
from realtime_trees_ros.payload_accumulator import PayloadAccumulator

def main(args=None):
    rclpy.init(args=args)
    node = PayloadAccumulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Node has been interrupted, shutting down...")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
