#!/usr/bin/env python3

from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import message_filters
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2 as pc2
import numpy as np
import queue
import time
import open3d as o3d
import multiprocessing as mp
from rclpy.serialization import serialize_message, deserialize_message

# ---------- helpers ----------

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    x, y, z, w = qx, qy, qz, qw
    xx = x * x; yy = y * y; zz = z * z
    xy = x * y; xz = x * z; yz = y * z
    wx = w * x; wy = w * y; wz = w * z
    R = np.array([
        [1 - 2 * (yy + zz),     2 * (xy - wz),         2 * (xz + wy)],
        [2 * (xy + wz),         1 - 2 * (xx + zz),     2 * (yz - wx)],
        [2 * (xz - wy),         2 * (yz + wx),         1 - 2 * (xx + yy)]
    ], dtype=np.float64)
    return R

def build_transform_from_odom(odom_msg):
    p = odom_msg.pose.pose.position
    o = odom_msg.pose.pose.orientation
    R = quaternion_to_rotation_matrix(o.x, o.y, o.z, o.w)
    t = np.array([p.x, p.y, p.z], dtype=np.float64)
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = t
    return M

def transform_points(M, points):
    if points.size == 0:
        return points.reshape(0, 3)
    R = M[:3, :3]
    t = M[:3, 3]
    return (points @ R.T) + t

def voxel_downsample_numpy(points, voxel_size):
    if points.shape[0] == 0 or voxel_size <= 0.0:
        return points
    scaled = np.floor(points / float(voxel_size)).astype(np.int64)
    _, unique_indices = np.unique(scaled, axis=0, return_index=True)
    return points[unique_indices]

def voxel_downsample_open3d(points, voxel_size):
    if points.shape[0] == 0 or voxel_size <= 0.0:
        return points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    down = pcd.voxel_down_sample(voxel_size)
    return np.asarray(down.points, dtype=np.float64)

def pc2_to_xyz_array_fast(msg: PointCloud2):
    pt_cnt = msg.width * msg.height
    if pt_cnt == 0:
        return np.zeros((0, 3), dtype=np.float32)
    
    if msg.is_bigendian:
        print(f"Fast conversion failed: bigendian format not supported (is_bigendian={msg.is_bigendian})")
        return None
    
    if len(msg.fields) < 3:
        print(f"Fast conversion failed: insufficient fields {len(msg.fields)}, expected at least 3")
        return None
    
    # Determine datatype from first field
    first_field_datatype = msg.fields[0].datatype
    if first_field_datatype == 7:  # FLOAT32
        np_dtype = np.float32
        bytes_per_field = 4
    elif first_field_datatype == 8:  # FLOAT64
        np_dtype = np.float64
        bytes_per_field = 8
    else:
        print(f"Fast conversion failed: unsupported datatype {first_field_datatype}, expected 7 (FLOAT32) or 8 (FLOAT64)")
        return None
    
    for i, name in enumerate(['x', 'y', 'z']):
        if i >= len(msg.fields):
            print(f"Fast conversion failed: missing field '{name}' at index {i}")
            return None
        f = msg.fields[i]
        if f.name != name:
            print(f"Fast conversion failed: field {i} name '{f.name}' != expected '{name}'")
            return None
        if f.datatype != first_field_datatype:
            print(f"Fast conversion failed: field '{name}' datatype {f.datatype} != expected {first_field_datatype}")
            return None
        if f.offset != i * bytes_per_field:
            print(f"Fast conversion failed: field '{name}' offset {f.offset} != expected {i * bytes_per_field}")
            return None
    
    try:
        arr = np.frombuffer(msg.data, dtype=np_dtype)
        arr = arr.reshape(pt_cnt, int(msg.point_step // bytes_per_field))
        return arr[:, :3]
    except Exception as e:
        print(f"Fast conversion failed during numpy operations: {e}")
        return None

def pc2_to_xyz_array_generic(msg: PointCloud2):
    print("WARNING: Using generic (slow) PointCloud2 to XYZ conversion")
    gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    arr = np.fromiter((coord for point in gen for coord in point), dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    arr = arr.reshape(-1, 3)
    return arr

def worker_process(job_q: mp.Queue, result_q: mp.Queue, voxel_size: float, filter_at_end: bool):
    while True:
        try:
            print("Worker waiting for job...")
            job = job_q.get()
            print("Worker got job")
            if job is None:
                break
            pcs_list, last_tf, frame_id, stamp_sec, stamp_nanosec, path_odom_list = job
            # Merge
            if not pcs_list:
                pts_map = np.zeros((0, 3), dtype=np.float64)
            else:
                pcs_list = [np.asarray(p, dtype=np.float64) for p in pcs_list]
                pts_map = np.vstack(pcs_list)
            # Filter
            if filter_at_end:
                pts_map = voxel_downsample_open3d(pts_map, voxel_size)
            # Build message and serialize for IPC
            header = Header()
            header.frame_id = frame_id
            header.stamp.sec = int(stamp_sec)
            header.stamp.nanosec = int(stamp_nanosec)
            cloud_msg = pc2.create_cloud_xyz32(header, pts_map.astype(np.float32).tolist())
            result_q.put((serialize_message(cloud_msg)))
            print("Worker done")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Worker encountered exception: {e}")
            continue

# ---------- Node ----------
class PayloadAccumulator(Node):
    def __init__(self):
        super().__init__('payload_accumulator')
        self.declare_parameter('topics.pointcloud_topic', '/points')
        self.declare_parameter('topics.odometry_topic', '/odom')
        self.declare_parameter('topics.posegraph_topic', '')
        self.declare_parameter('accumulation.type', 'both')
        self.declare_parameter('accumulation.distance', 20.0)
        self.declare_parameter('accumulation.time', 20.0)
        self.declare_parameter('processing.voxel_size', 0.05)
        self.declare_parameter('processing.posegraph_interval', 0.5)
        self.declare_parameter('filter_at_end', True)
        self.declare_parameter('sync_slop', 0.05)
        self.declare_parameter('sync_queue_size', 10)

        self.pointcloud_topic = self.get_parameter('topics.pointcloud_topic').get_parameter_value().string_value
        self.odom_topic = self.get_parameter('topics.odometry_topic').get_parameter_value().string_value
        self.posegraph_topic = self.get_parameter('topics.posegraph_topic').get_parameter_value().string_value
        self.accumulation_type = self.get_parameter('accumulation.type').get_parameter_value().string_value
        self.max_distance_m = self.get_parameter('accumulation.distance').get_parameter_value().double_value
        self.max_time_s = self.get_parameter('accumulation.time').get_parameter_value().double_value
        self.voxel_size = self.get_parameter('processing.voxel_size').get_parameter_value().double_value
        self.posegraph_interval = self.get_parameter('processing.posegraph_interval').get_parameter_value().double_value
        self.filter_at_end = self.get_parameter('filter_at_end').get_parameter_value().bool_value
        self.sync_slop = self.get_parameter('sync_slop').get_parameter_value().double_value
        self.sync_queue_size = int(self.get_parameter('sync_queue_size').get_parameter_value().integer_value)
        self.payload_topic = '/local_mapping/payload_in_local'  # as used in realtime_trees
        self.path_topic = '/local_mapping/path'  # as used in realtime_trees
        
        self.get_logger().info(f"PayloadAccumulator configured with pointcloud_topic='{self.pointcloud_topic}', odometry_topic='{self.odom_topic}', posegraph_topic='{self.posegraph_topic}', accumulation_type='{self.accumulation_type}', max_distance_m={self.max_distance_m}, max_time_s={self.max_time_s}, voxel_size={self.voxel_size}, posegraph_interval={self.posegraph_interval}, filter_at_end={self.filter_at_end}")
        self.get_logger().info(f"PayloadAccumulator publishing to payload_topic='{self.payload_topic}', path_topic='{self.path_topic}'")
        self.get_logger().info(f"PayloadAccumulator using sync_slop={self.sync_slop}, sync_queue_size={self.sync_queue_size}")
        

        qos_pc = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        qos_odom = QoSProfile(depth=50)
        
        self.pc_mf = message_filters.Subscriber(self, PointCloud2, self.pointcloud_topic, qos_profile=qos_pc)
        self.odom_mf = message_filters.Subscriber(self, Odometry, self.odom_topic, qos_profile=qos_odom)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.pc_mf, self.odom_mf], self.sync_queue_size, self.sync_slop)
        self.ts.registerCallback(self.sync_callback)

        self.accum_list = []
        self.payload_start_time = None
        self.path_buffer = []  # Buffer to store odometry messages for path publishing
        self.posegraph_buffer = []  # Buffer to store posegraph messages if needed
        self.last_posegraph_stamp = 0.0  # Track last posegraph insertion timestamp from odometry header

        self.last_odom_pos = None
        self.path_length = 0.0

        self.last_transform_used = None
        self.last_body_frame = ''
        self.last_cloud_header = None

        self.job_q = mp.Queue(maxsize=2)
        self.result_q = mp.Queue(maxsize=2)
        self.worker_p = mp.Process(target=worker_process, args=(self.job_q, self.result_q, self.voxel_size, self.filter_at_end), daemon=True)
        self.worker_p.start()
        self.result_timer = self.create_timer(0.02, self.poll_results)

        self.payload_pub = self.create_publisher(PointCloud2, self.payload_topic, 1)
        self.path_pub = self.create_publisher(Path, self.path_topic, 1)
        if self.posegraph_topic:
            self.posegraph_pub = self.create_publisher(Path, self.posegraph_topic, 1)

    def sync_callback(self, pc_msg: PointCloud2, odom_msg: Odometry):
        self.map_frame_id = odom_msg.header.frame_id if odom_msg.header.frame_id else 'map'
        debug_string = ""
        t0 = time.perf_counter()
        xyz = pc2_to_xyz_array_fast(pc_msg)
        if xyz is None:
            xyz = pc2_to_xyz_array_generic(pc_msg)
        if getattr(xyz, 'base', None) is not None:
            xyz = xyz.copy()

        M = build_transform_from_odom(odom_msg)
        pts_map = transform_points(M, xyz.astype(np.float64))
        t1 = time.perf_counter()
        debug_string += f"sync_callback: pc2_to_xyz_array {1000*(t1-t0):.2f} ms"

        stamp = pc_msg.header.stamp.sec + pc_msg.header.stamp.nanosec * 1e-9
        body_frame = pc_msg.header.frame_id
        header_copy = Header()
        header_copy.stamp = pc_msg.header.stamp
        header_copy.frame_id = body_frame
        # Track last transform/frame/header for job building
        self.last_transform_used = M
        self.last_body_frame = body_frame
        self.last_cloud_header = header_copy

        pos = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z], dtype=np.float64)
        if self.last_odom_pos is None:
            self.last_odom_pos = pos
            self.path_length = 0.0
        else:
            delta = np.linalg.norm(pos - self.last_odom_pos)
            self.path_length += float(delta)
            self.last_odom_pos = pos
        t2 = time.perf_counter()
        debug_string += f", odom transform {1000*(t2-t1):.2f} ms"

        if not self.accum_list:
            self.payload_start_time = stamp
        self.accum_list.append(pts_map)
        self.path_buffer.append(odom_msg)  # Add odometry message to path buffer
        elapsed = stamp - self.payload_start_time if self.payload_start_time is not None else 0.0
        traveled = self.path_length
        self.publish_buffer_as_path(self.path_buffer, self.path_pub)
        
        # Add current pose to posegraph buffer based on odometry timestamp
        if self.posegraph_topic:
            odom_stamp = odom_msg.header.stamp.sec + odom_msg.header.stamp.nanosec * 1e-9
            if odom_stamp - self.last_posegraph_stamp >= self.posegraph_interval:
                self.posegraph_buffer.append(odom_msg)
                self.last_posegraph_stamp = odom_stamp
                self.publish_buffer_as_path(self.posegraph_buffer, self.posegraph_pub)
        
        t3 = time.perf_counter()
        debug_string += f", accumulation {1000*(t3-t2):.2f} ms"

        # Check accumulation type and trigger conditions
        should_publish = False
        if self.accumulation_type == 'distance':
            should_publish = traveled >= self.max_distance_m
        elif self.accumulation_type == 'time':
            should_publish = elapsed >= self.max_time_s
        elif self.accumulation_type == 'both':
            should_publish = traveled >= self.max_distance_m or elapsed >= self.max_time_s
        
        if should_publish:
            # Send the list of individual clouds to the worker process
            pcs_list = self.accum_list
            path_odom_list = self.path_buffer.copy()  # Copy the path buffer for worker
            last_tf = self.last_transform_used.copy() if self.last_transform_used is not None else M.copy()
            last_frame = self.map_frame_id
            last_header = self.last_cloud_header or header_copy
            t4 = time.perf_counter()
            debug_string += f", job prep {1000*(t4-t3):.2f} ms"
            
            self.accum_list = []
            self.path_buffer = []  # Reset path buffer
            self.payload_start_time = None
            self.path_length = 0.0
            
            # Add current pose to posegraph buffer when payload is sent to worker
            self.posegraph_buffer.append(odom_msg)
            self.publish_buffer_as_path(self.posegraph_buffer, self.posegraph_pub)
            
            try:
                self.job_q.put_nowait((pcs_list, last_tf, last_frame, int(last_header.stamp.sec), int(last_header.stamp.nanosec), path_odom_list))
                t5 = time.perf_counter()
                debug_string += f", queue put {1000*(t5-t4):.2f} ms"
            except queue.Full:
                t5 = time.perf_counter()
                self.get_logger().warn(f'Worker queue full, dropping payload job (queue attempt: {1000*(t5-t4):.2f} ms)')
        else:
            t4 = time.perf_counter()
            debug_string += f", no job submission (conditions not met) {1000*(t4-t2):.2f} ms"
        
        t_final = time.perf_counter()
        debug_string += f", total callback {1000*(t_final-t0):.2f} ms"
        # self.get_logger().error(f"We are accumulating {len(self.accum_list)} clouds, elapsed={elapsed:.2f} s. This results in an average pc processing rate of {len(self.accum_list)/(elapsed+0.1) if elapsed>0 else 0:.1f} Hz")
        # self.get_logger().error(debug_string)

    def publish_buffer_as_path(self, buffer: list[Odometry], publisher: Publisher):
        path_msg = Path()
        if buffer:
            path_msg.header.stamp = buffer[-1].header.stamp
            path_msg.header.frame_id = buffer[0].header.frame_id
            path_msg.poses = []
            for odom_msg in buffer:
                pose_stamped = PoseStamped()
                pose_stamped.header = odom_msg.header
                pose_stamped.pose = odom_msg.pose.pose
                path_msg.poses.append(pose_stamped)
            publisher.publish(path_msg)

    def poll_results(self):
        processed = 0
        while processed < 3:  # limit per timer tick
            try:
                data = self.result_q.get_nowait()
            except queue.Empty:
                break
            try:
                cloud_msg = deserialize_message(data, PointCloud2)
                self.payload_pub.publish(cloud_msg)
            except Exception as e:
                self.get_logger().error(f"Failed to deserialize/publish payload: {e}")
            processed += 1

    # Ensure worker process is terminated on node destruction
    def destroy_node(self):
        try:
            self.job_q.put_nowait(None)
        except Exception:
            pass
        try:
            if hasattr(self, 'worker_p') and self.worker_p is not None:
                self.worker_p.join(timeout=1.0)
                if self.worker_p.is_alive():
                    self.worker_p.terminate()
        except Exception:
            pass
        return super().destroy_node()
