import gc
from typing import List, Union

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from builtin_interfaces.msg import Time as BuiltinTime
from geometry_msgs.msg import PoseStamped, Point, Pose, Quaternion
from realtime_trees.circle import Circle
from realtime_trees.clustering import cluster
from realtime_trees.terrain import fit_terrain
from realtime_trees.tree import Tree
from realtime_trees.utils.matrix_calc import efficient_inv, T_to_pos_and_quat
from realtime_trees.utils.ros_proxys import Time, StampedPose
from realtime_trees.utils.timing import Timer
from realtime_trees.utils.dataclasses import Cluster
from realtime_trees_msgs.msg import Circle as CircleMsg, Tree as TreeMsg
from rclpy.duration import Duration
from rclpy.time import Time as RclpyTime
from sensor_msgs.msg import PointCloud2
from shape_msgs.msg import Mesh as MeshMsg, MeshTriangle
from visualization_msgs.msg import Marker


def rospy_pose2T(orientation, position) -> np.ndarray:
    """Converts a ros pose to a 4x4 transformation matrix

    Args:
        orientation (rospy orientation): ros orientation with quaternions (x, y, z, w)
        position (rospy position): ros position

    Returns:
        np.ndarray: 4x4 transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(
        np.array(
            [
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            ]
        )
    ).as_matrix()
    T[:3, 3] = np.array([position.x, position.y, position.z])
    return T


def pc2_to_np(msg: PointCloud2):
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
        arr = np.frombuffer(msg.data, dtype=np_dtype).copy()  # copy to ensure memory is owned
        arr = arr.reshape(pt_cnt, int(msg.point_step // bytes_per_field))
        return arr[:, :3]
    except Exception as e:
        print(f"Fast conversion failed during numpy operations: {e}")
        return None


def pc2_to_o3d(cloud: PointCloud2) -> o3d.t.geometry.PointCloud:
    """Helper function to convert a PointCloud2 message to an open3d point cloud

    Args:
        cloud (PointCloud2): message with point cloud data

    Returns:
        o3d.t.geometry.PointCloud: output o3d point cloud
    """
    # Point step in the following examples is generally 48 and therefore for float32: 12
    cloud_numpy = pc2_to_np(cloud)
    cloud = o3d.t.geometry.PointCloud(cloud_numpy[:, :3])
    # Only set normals if we have enough columns
    if cloud_numpy.shape[1] >= 7:
        cloud.point.normals = cloud_numpy[:, 4:7]
    else:
        timer_tmp = Timer()
        with timer_tmp("normal_estimation"):
            cloud.estimate_normals()
        print(timer_tmp)
            
    return cloud


def radius_crop_pc(
    cloud: o3d.t.geometry.PointCloud,
    center_pose: np.ndarray,
    radius: float,
) -> o3d.t.geometry.PointCloud:
    """Crops a point cloud using a max distance from the sensor. The sensor pose

    Args:
        cloud (o3d.t.geometry.PointCloud): Point cloud to be cropped
        sensor_pose (np.ndarray): 4x4 transformation matrix from sensor to odom
        radius (float): maximum distance from sensor in m

    Returns:
        o3d.t.geometry.PointCloud: output o3d point cloud
    """
    # Transform the point cloud to sensor frame
    cloud = cloud.transform(efficient_inv(center_pose))
    # Calculate the distance from the sensor
    distances = np.linalg.norm(cloud.point.positions.numpy()[:, :2], axis=1)
    cloud = cloud.select_by_mask(distances <= radius)
    cloud = cloud.transform(center_pose)

    return cloud


def clustering_worker_fun(
    tf_buffer,
    cloud_msg,
    terrain_enabled,
    crop_radius,
    path_odom,
    pose_graph_stamps,
    crop_bounds,
    max_cluster_radius,
    precise_calcuation_fraction,
    debug_level,
    map_frame_id,
    odom_frame_id,
    normal_thr,
    voxel_size,
    **kwargs,
) -> Union[List[Cluster], np.ndarray, Timer]:
    """This is a helper function to offload the clusterin of the worker into separate
    threads. This must no be a member function of a class, which is why it is defined
    here.

    Args:
        transferred from the main thread

    Returns:
        Union[List[Cluster], np.ndarray, Timer]: clusters, terrain, and timer of clustering worker
    """
    try:
        print("Cluster worker called")
        if pose_graph_stamps is None:
            print(
                "pose_graph_stamps is empty, maybe restarted node amid playback?"
            )
            return

        # find index closest to center of path that also is in posegraph
        center_index = len(path_odom.poses) // 2
        center_index_found = False
        for i, step_size in enumerate(range(len(path_odom.poses))):
            center_index += step_size * (-1) ** i
            if path_odom.poses[center_index].header.stamp in pose_graph_stamps:
                center_index_found = True
                break
        if not center_index_found:
            print("Could not find any cloud's stamp in posegraph")
            return

        center_pose = path_odom.poses[center_index].pose
        center_stamp = path_odom.poses[center_index].header.stamp
        T_sensor2odom = rospy_pose2T(center_pose.orientation, center_pose.position)
        pose_odom2map = tf_buffer.lookup_transform(
            map_frame_id,
            odom_frame_id,
            Duration(seconds=0),
            Duration(seconds=1),
        )
        T_odom2map = rospy_pose2T(
            pose_odom2map.transform.rotation, pose_odom2map.transform.translation
        )
        T_sensor2map = T_odom2map @ T_sensor2odom

        timer = Timer()
        with timer("cw"):
            with timer("cw/conversion"):
                cloud = pc2_to_o3d(cloud_msg)
                cloud = radius_crop_pc(
                    cloud,
                    center_pose=T_sensor2odom,
                    radius=crop_radius,
                )
            if terrain_enabled is not None:
                with timer("cw/terrain"):
                    terrain = fit_terrain(cloud)
            else:
                terrain = None
            with timer("cw/clustering"):
                clusters = cluster(
                    cloud=cloud,
                    terrain=terrain,
                    crop_bounds=crop_bounds,
                    max_cluster_radius=max_cluster_radius,
                    precise_calcuation_fraction=precise_calcuation_fraction,
                    debug_level=debug_level,
                    normal_thr=normal_thr,
                    voxel_size=voxel_size,
                )
            # convert clusters into stamped sensor frame
            for i in range(len(clusters)):
                clusters[i].cloud.transform(efficient_inv(T_sensor2odom))
                clusters[i].info.axis.transform = (
                    efficient_inv(T_sensor2odom)
                    @ clusters[i].info.axis.transform
                )
                clusters[i].info.T_sensor2map = T_sensor2map
                clusters[i].info.time_stamp = builtin_time_to_time(center_stamp)
        gc.collect()
    except Exception as e:
        # if not done this way, the exception is not printed in the different thread
        print(f"Exception in cluster worker: {e.with_traceback(e.__traceback__)}")
        return
    return clusters, terrain, timer


def set_axes_equal(ax):
    # from https://stackoverflow.com/questions/13685386/how-to-set-the-equal-aspect-ratio-for-all-axes-x-y-z
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def generate_mesh_msg(
    vertices: np.ndarray,
    triangles: np.ndarray,
    id: int = None,
    frame_id: str = None,
    time_stamp=None,
    color=[140 / 255.0, 102 / 255.0, 87 / 255.0],
    alpha=1.0,
) -> Marker:
    """Helper function to generate a mesh message

    Args:
        vertices (np.ndarray): vertices of mesh
        triangles (np.ndarray): triangles of mesh
        id (int, optional): unique id of mesh. Defaults to None.
        frame_id (str, optional): frame in which the mesh is represented.
            Defaults to None.
        time_stamp (_type_, optional): time stamp of the mesh. Defaults to None.
        color (list, optional): color of the mesh. Defaults to
            [140 / 255.0, 102 / 255.0, 87 / 255.0].
        alpha (float, optional): transparency of the mesh. Defaults to 1.0.

    Returns:
        Marker: _description_
    """
    if vertices.shape[0] == 0:
        return None

    mesh_msg = Marker()
    mesh_msg.header.frame_id = frame_id
    mesh_msg.header.stamp = time_stamp
    mesh_msg.ns = "realtime_trees/meshes"
    mesh_msg.id = id if id is not None else np.random.randint(0, 100000)
    mesh_msg.type = Marker.TRIANGLE_LIST
    mesh_msg.action = Marker.ADD

    mesh_msg.pose.orientation.w = 1.0
    mesh_msg.scale.x = 1.0
    mesh_msg.scale.y = 1.0
    mesh_msg.scale.z = 1.0
    mesh_msg.color.a = alpha
    mesh_msg.color.r = color[0]
    mesh_msg.color.g = color[1]
    mesh_msg.color.b = color[2]

    mesh_msg.points = [
        Point(x=x, y=y, z=z) for x, y, z in vertices[triangles].reshape(-1, 3).tolist()
    ]

    return mesh_msg


def rclpy_time_to_time(rclpy_time: RclpyTime) -> Time:
    """Helper function to convert a rclpy time object to a proxy time that can be used
    with the rest of the realtime_trees code

    Args:
        rospy_time (rclpy.time.Time): ros time representation

    Returns:
        Time: realtime_trees time represenation
    """
    seconds, nanoseconds = rclpy_time.seconds_nanoseconds()
    return Time(seconds=seconds, nanoseconds=nanoseconds)


def builtin_time_to_time(builtin_time: BuiltinTime) -> Time:
    """Helper function to convert a ROS 2 built-in time object to a proxy time that can be used
    with the rest of the realtime_trees code

    Args:
        rospy_time (builtin_interfaces.msg.Time): ROS 2 message time representation

    Returns:
        Time: realtime_trees time represenation
    """
    return Time(seconds=builtin_time.sec, nanoseconds=builtin_time.nanosec)


def rospy_stamped_pose_to_stamped_pose(stamped_pose: PoseStamped) -> StampedPose:
    """Helper function to convert a rospy stamped pose to a proxy stamped pose that can
    be used with the rest of the realtime_trees code

    Args:
        stamped_pose (PoseStamped): ros stamped pose

    Returns:
        StampedPose: realtime_trees stamped pose
    """
    return StampedPose(
        rospy_pose2T(stamped_pose.pose.orientation, stamped_pose.pose.position),
        builtin_time_to_time(stamped_pose.header.stamp)
    )


def circle_to_circle_msg(
    circle: Circle, frame_id: str, stamp: RclpyTime
) -> CircleMsg:
    """Helper function to convert a circle object to a circle message that can be used
    build a tree message or to communicate circle measurements to other ROS nodes.

    Args:
        circle (Circle): Circle object to be converted
        frame_id (str): The frame ID for the circle message.
        stamp (rclpy.time.Time): The timestamp for the circle message.
            Defaults to None.

    Returns:
        CircleMsg: ROS message described in this package
    """
    circle_msg = CircleMsg()
    circle_msg.header.frame_id = frame_id
    circle_msg.header.stamp = stamp
    circle_msg.radius = circle.radius
    circle_msg.pose.position.x = circle.center[0]
    circle_msg.pose.position.y = circle.center[1]
    circle_msg.pose.position.z = circle.center[2]
    quat = Rotation.from_matrix(circle.rot_mat).as_quat()
    circle_msg.pose.orientation.x = quat[0]
    circle_msg.pose.orientation.y = quat[1]
    circle_msg.pose.orientation.z = quat[2]
    circle_msg.pose.orientation.w = quat[2]
    return circle_msg


def tree_to_tree_msg(
    tree: Tree,
    coverage_angle: float,
    frame_id: str,
    stamp: BuiltinTime,
) -> TreeMsg:
    """Helper function to convert a tree object to a tree message that can be used to
    communicate the measurements to other ROS nodes.

    Args:
        tree (Tree): Tree to be converted
        coverage_angle (float): Coverage angle if applicable. As this is a
            metric not akin to a tree, it is not stored in the tree object itself.
        frame_id (str): The frame ID for the tree message.
        stamp (rclpy.time.Time): The timestamp for the tree message. 
            If not provided, the current time will be used. Dafaults to None.

    Returns:
        TreeMsg: ROS message described in this package or None if tree is not 
            reconstructed
    """
    # if not tree.reconstructed:
    #     return None
    tree_msg = TreeMsg()
    tree_msg.header.frame_id = frame_id
    tree_msg.header.stamp = stamp
    tree_msg.id = tree.id
    pos, quat = T_to_pos_and_quat(tree.axis.transform)
    tree_msg.pose.position.x = pos[0]
    tree_msg.pose.position.y = pos[1]
    tree_msg.pose.position.z = pos[2]
    tree_msg.pose.orientation.x = quat[0]
    tree_msg.pose.orientation.y = quat[1]
    tree_msg.pose.orientation.z = quat[2]
    tree_msg.pose.orientation.w = quat[3]
    if tree.reconstructed:
        tree_msg.circles = [circle_to_circle_msg(circle, frame_id, stamp) for circle in tree.circles]
    else:
        tree_msg.circles = []
    if tree.canopy_mesh is not None:
        tree_msg.canopy_mesh = mesh_to_mesh_msg(
            tree.canopy_mesh["vertices"], tree.canopy_mesh["triangles"]
        )
    tree_msg.coverage_angle = coverage_angle if coverage_angle else -1.0
    tree_msg.dbh = tree.dbh if tree.dbh else 2 * tree.axis.radius
    tree_msg.canopy_volume = tree.canopy_volume if tree.canopy_volume else -1.0
    tree_msg.reconstructed = tree.reconstructed
    return tree_msg


def mesh_to_mesh_msg(vertices: np.ndarray, triangles: np.ndarray) -> MeshMsg:
    """Converts a mesh given as np.ndarrays of vertices and triangles into a
    shape_msgs/Mesh message.

    Args:
        vertices (np.ndarray): Vertices of the mesh
        triangles (np.ndarray): Triangles of the mesh

    Returns:
        MeshMsg: shape_msgs/Mesh message
    """
    mesh_msg = MeshMsg()
    mesh_msg.vertices = [Point(x=x, y=y, z=z) for x, y, z in vertices.tolist()]
    mesh_msg.triangles = [MeshTriangle(vertex_indices=[a, b, c]) for a, b, c in triangles.tolist()]
    return mesh_msg
