# author: Leonard Freissmuth

import colorsys
from functools import partial
import gc
from multiprocessing.pool import ThreadPool
import os
import pickle
from typing import List, Union
from threading import Lock
import yaml

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from builtin_interfaces.msg import Time as BuiltinTime
import message_filters
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_srvs.srv import Empty
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from rclpy.time import Time
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from realtime_trees.utils.meshing import meshgrid_to_mesh
from realtime_trees.utils.timing import Timer
from realtime_trees.tree_manager import TreeManager
from realtime_trees_msgs.msg import TreeManager as TreeManagerMsg
from realtime_trees_msgs.srv import CustomString
from std_msgs.msg import Header
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray

from realtime_trees_ros.utils import (
    clustering_worker_fun,
    mesh_to_mesh_msg,
    rospy_stamped_pose_to_stamped_pose,
    generate_mesh_msg,
    tree_to_tree_msg,
)
from realtime_trees.utils.dataclasses import Cluster


timer = Timer()
# ignore divide by zero warnings, code actively works with nans
np.seterr(divide="ignore", invalid="ignore")


class RealtimeTrees(Node):
    def __init__(self) -> None:
        """This ForestAnalysis class is the main class for the realtime tree analysis
        in the ROS setting. It listens to point cloud signals and path signals and
        issues reconstruction and clustering of trees. It also listens to posegraph
        updates to update the tree poses.
        """
        super().__init__('realtime_trees_ros')

        # ROS parameters
        self.get_params()

        self.get_logger().info(f"Output directory set to '{self.base_output_path}'.")
        os.makedirs(self.base_output_path, exist_ok=True)

        self.path_accumulators = [
            {"state": "active", "path": None},
            {"state": "empty", "path": None}
        ] # states can be "empty", "ready", and "active" (if currently being filled)
        self.prev_path_len = -1

        self.last_pc_header = None
        self.pc_counter = 0
        self.tree_manager_lock = Lock()
        self.pose_graph_stamps = []
        self.posegraph_updates_active = True
        self.visibility_mask = None

        self._clustering_pool = ThreadPool(processes=self._clustering_n_threads)
        reconstruction_args = {
            "max_height": self._fitting_max_height,
            "slice_heights": self._fitting_slice_heights,
            "slice_thickness": self._fitting_slice_thickness,
            "max_center_deviation": self._fitting_max_center_deviation,
            "max_radius": self._max_reco_diameter / 2,
            "filter_radius": self._fitting_filter_radius,
            "max_consecutive_fails": self._fitting_max_consecutive_fails,
        }

        self._tree_manager = TreeManager(
            self._distance_threshold,
            self._reco_min_angle_coverage,
            self._reco_min_distance,
            self._terrain_confidence_stds,
            self._terrain_confidence_sensor_weight,
            self._terrain_use_embree,
            self._generate_canopy_mesh,
            output_path=self.base_output_path,
            debug_level=self._debug_level,
            payload_crop_radius=self._clustering_crop_radius,
            reconstruction_args=reconstruction_args,
        )

        # ROS publishers and subscribers
        self.init_ros()
        return

    def get_params(self):
        """Reads the parameters from the ROS parameter server and sets the class
        variables accordingly
        """
        self.declare_parameter('debug_level', 0, descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER
            )
        )
        self._debug_level =  self.get_parameter('debug_level').get_parameter_value().integer_value

        # Frames
        self.declare_parameter('frames.base_frame_id', 'base', descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING
            )
        )
        self._base_frame_id =  self.get_parameter('frames.base_frame_id').get_parameter_value().string_value

        self.declare_parameter('frames.odom_frame_id', 'odom', descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING
            )
        )
        self._odom_frame_id = self.get_parameter('frames.odom_frame_id').get_parameter_value().string_value

        self.declare_parameter('frames.map_frame_id', 'map', descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING
            )
        )
        self._map_frame_id = self.get_parameter('frames.map_frame_id').get_parameter_value().string_value

        # Tree Manager
        self.declare_parameter('tree_manager.distance_threshold', 0.5, descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE
            )
        )
        self._distance_threshold = self.get_parameter('tree_manager.distance_threshold').get_parameter_value().double_value

        self.declare_parameter('tree_manager.reco_min_angle_coverage', 180.0, descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE
            )
        )
        self._reco_min_angle_coverage = np.deg2rad(
            self.get_parameter('tree_manager.reco_min_angle_coverage').get_parameter_value().double_value
        )

        self.declare_parameter('tree_manager.reco_min_distance', 4.0, descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE
            )
        )
        self._reco_min_distance = self.get_parameter(
            'tree_manager.reco_min_distance'
        ).get_parameter_value().double_value

        self.declare_parameter('tree_manager.confidence_stds', [3, 5, 5], descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE_ARRAY
            )
        )
        self._terrain_confidence_stds = self.get_parameter(
            'tree_manager.confidence_stds'
        ).get_parameter_value().double_array_value

        self.declare_parameter('tree_manager.confidence_sensor_weight', 0.9999, descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE
            )
        )
        self._terrain_confidence_sensor_weight = self.get_parameter(
            'tree_manager.confidence_sensor_weight'
        ).get_parameter_value().double_value

        self.declare_parameter('tree_manager.use_embree', True, descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL
            )
        )
        self._terrain_use_embree = self.get_parameter('tree_manager.use_embree').get_parameter_value().bool_value
        self.declare_parameter('tree_manager.posegraph_updates_active', False, descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL
            )
        )
        self.posegraph_updates_active = self.get_parameter('tree_manager.posegraph_updates_active').get_parameter_value().bool_value

        # Terrain Fitting
        self.declare_parameter('terrain.enabled', True, descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL
            )
        )
        self._terrain_enabled = self.get_parameter('terrain.enabled').get_parameter_value().bool_value

        self.declare_parameter('terrain_fitting.smoothing', False, descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL
            )
        )
        self._terrain_smoothing = self.get_parameter('terrain_fitting.smoothing').get_parameter_value().bool_value

        self.declare_parameter('terrain.cell_size', 1, descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER
            )
        )
        self._terrain_cloth_cell_size = self.get_parameter('terrain.cell_size').get_parameter_value().integer_value

        # Clustering
        self.declare_parameter('clustering.crop_radius', 40.0, descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE
            )
        )
        self._clustering_crop_radius = self.get_parameter('clustering.crop_radius').get_parameter_value().double_value

        self.declare_parameter('clustering.crop_bounds', "[[0.5, 1.5], [2, 3], [4, 5]]", descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING
            )
        )
        self._clustering_crop_bounds = yaml.safe_load(self.get_parameter('clustering.crop_bounds').get_parameter_value().string_value)

        self.declare_parameter('clustering.max_cluster_radius', 5.0, descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE
            )
        )
        self._clustering_max_cluster_radius = self.get_parameter('clustering.max_cluster_radius').get_parameter_value().double_value

        self.declare_parameter('clustering.n_threads', 4, descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER
            )
        )
        self._clustering_n_threads = self.get_parameter('clustering.n_threads').get_parameter_value().integer_value

        self.declare_parameter('clustering.distance_calc_point_fraction', 0.1, descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE
            )
        )
        self._clustering_distance_calc_point_fraction = self.get_parameter('clustering.distance_calc_point_fraction').get_parameter_value().double_value

        self.declare_parameter('clustering.normal_thr', 0.5, descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE
            )
        )
        self._clustering_normal_thr = self.get_parameter('clustering.normal_thr').get_parameter_value().double_value

        self.declare_parameter('clustering.voxel_size', 0.05, descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE
            )
        )
        self._clustering_voxel_size = self.get_parameter('clustering.voxel_size').get_parameter_value().double_value

        # Fitting
        self.declare_parameter('fitting.max_height', 15.0, descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE
            )
        )
        self._fitting_max_height = self.get_parameter('fitting.max_height').get_parameter_value().double_value

        self.declare_parameter('fitting.slice_heights', 0.5, descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE
            )
        )
        self._fitting_slice_heights = self.get_parameter('fitting.slice_heights').get_parameter_value().double_value

        self.declare_parameter('fitting.slice_thickness', 0.3, descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE
            )
        )
        self._fitting_slice_thickness = self.get_parameter('fitting.slice_thickness').get_parameter_value().double_value

        self.declare_parameter('fitting.max_center_deviation', 0.05, descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE
            )
        )
        self._fitting_max_center_deviation = self.get_parameter('fitting.max_center_deviation').get_parameter_value().double_value

        self.declare_parameter('fitting.max_diameter', np.inf, descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE
            )
        )
        self._max_reco_diameter = self.get_parameter('fitting.max_diameter').get_parameter_value().double_value

        self.declare_parameter('fitting.filter_radius', 0.05, descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE
            )
        )
        self._fitting_filter_radius = self.get_parameter('fitting.filter_radius').get_parameter_value().double_value

        self.declare_parameter('fitting.max_consecutive_fails', 3, descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER
            )
        )
        self._fitting_max_consecutive_fails = self.get_parameter('fitting.max_consecutive_fails').get_parameter_value().integer_value

        self.declare_parameter('fitting.generate_canopy', True, descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL
            )
        )
        self._generate_canopy_mesh = self.get_parameter('fitting.generate_canopy').get_parameter_value().bool_value

        # Output
        self.declare_parameter('output.path', '/tmp/realtime_trees',
            descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING),
        )
        self.base_output_path = self.get_parameter('output.path').get_parameter_value().string_value

        return

    def init_ros(self) -> None:
        """Sets up all the ROS related components like the publishers, subscribers and
        services
        """

        # listeners for transforms
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Services
        self.export_tree_manager_service = self.create_service(
            Empty,
            "~/export_tree_manager",
            self.export_tree_manager_callback,
        )

        self.toggle_tree_service = self.create_service(
            Empty,
            "~/toggle_tree_visibility",
            self.toggle_tree_visibility_callback,
        )

        self.toggle_loop_closure_service = self.create_service(
            Empty,
            "~/toggle_loop_closure",
            self.toggle_loop_closure_callback,
        )

        # Subscribers
        self._sub_posegraph_update = self.create_subscription(
            Path,
            "/pose_graph",
            self.posegraph_changed_callback,
            QoSProfile(depth=10)
        )
        self._sub_payload_cloud = message_filters.Subscriber(
            self, PointCloud2, "/local_mapping/payload_in_local"
        )
        self._sub_payload_info = message_filters.Subscriber(
            self, Path, "/local_mapping/path"
        )
        self._sub_payload_cloud.registerCallback(self.payload_callback)
        self._sub_payload_info.registerCallback(self.path_callback)

        # Publishers
        qos_profile = QoSProfile(depth=1)
        qos_profile.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self._pub_stem_meshes = self.create_publisher(
            MarkerArray, "~/stem_meshes", qos_profile
        )
        self._pub_debug_clusters = self.create_publisher(
            PointCloud2, "~/debug_clusters", qos_profile
        )
        self._pub_terrain_model = self.create_publisher(
            Marker, "~/terrain_model", qos_profile
        )
        self._pub_canopy_meshes = self.create_publisher(
            MarkerArray, "~/canopy_meshes", qos_profile
        )
        self._pub_tree_clusters = self.create_publisher(
            PointCloud2, "~/tree_point_clouds", qos_profile
        )
        self._pub_cluster_labels = self.create_publisher(
            MarkerArray, "~/tree_labels", qos_profile
        )
        self._pub_tree_manager = self.create_publisher(
            TreeManagerMsg, "~/tree_manager", qos_profile
        )
        return

    def publish_pointclouds(
        self,
        pub,
        clouds: list,
        colors: list = None,
        frame_id=None,
        time_stamp=None,
    ):
        """Helper function to publish a set of point clouds

        Args:
            pub (rospy.Publisher): publisher used for publishing
            clouds (list): list of point clouds to publish
            colors (list, optional): list of colors for each point clodu.
                Defaults to None.
            frame_id (_type_, optional): frame the point clouds are represented in.
                Defaults to None.
            time_stamp (_type_, optional): time stamp of the point clouds.
                Defaults to None.
        """
        # convert colors to single float32
        if not colors:
            colors = [np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)] * len(clouds)
        for i, color in enumerate(colors):
            color = np.floor(np.array([color[2], color[1], color[0], 0.0]) * 255)
            color = np.frombuffer(color.astype(np.uint8).tobytes(), dtype=np.float32)
            colors[i] = color

        # Convert numpy arrays to pointcloud2 data
        header = Header()
        header.frame_id = frame_id if frame_id else self.last_pc_header.frame_id
        header.stamp = time_stamp if time_stamp else self.last_pc_header.stamp
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1)
        ]

        points = np.concatenate(
            [
                np.hstack([cloud, np.repeat(color, cloud.shape[0])[:, None]]).astype(
                    np.float32
                )
                for cloud, color in zip(clouds, colors)
            ]
        )

        # Publish the pointcloud
        cloud = point_cloud2.create_cloud(header, fields, points)
        pub.publish(cloud)
        if self._debug_level > 0:
            self.get_logger().info(f"Published {len(clouds)} colored pointclouds!")
        return

    def publish_cluster_labels(
        self, labels: List[str], locations: List[np.ndarray], frame_id=None
    ):
        """Helper function to publish a set of strings at given positions, which can be
        used to label points.

        Args:
            labels (List[str]): list of label texts
            locations (List[np.ndarray]): list of lable locations
            frame_id (_type_, optional): frame id the label location is represented in.
                Defaults to None.
        """
        marker_array = MarkerArray()
        for i, (label, location) in enumerate(zip(labels, locations)):
            marker = Marker()
            marker.header.frame_id = (
                frame_id if frame_id else self.last_pc_header.frame_id
            )
            marker.header.stamp = self.last_pc_header.stamp
            marker.ns = "realtime_trees/markers"
            marker.id = i
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD

            marker.pose.position.x = location[0]
            marker.pose.position.y = location[1]
            marker.pose.position.z = location[2]
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.text = label
            marker_array.markers.append(marker)
        self._pub_cluster_labels.publish(marker_array)
        return

    def _clustering_worker_finished_callback(
        self,
        result: Union[List[Cluster], np.ndarray, Timer],
        path_odom: np.ndarray,
        pc_counter: int,
    ):
        """Callback for when the clustering worker (on a different thread) is finished

        Args:
            result (Union[List[Cluster], np.ndarray, Timer]): result of clustering worker
            path_odom (np.ndarray): ath of the sensor in the odom frame
            pc_counter (int): counter of the point cloud (just for print outs)
        """
        self.get_logger().info("clustering worker finished callback")
        if result is None:
            self.get_logger().info("clustering worker has no valid result")
            return
        clusters, terrain, timer_cw = result
        timer_cwc = Timer()
        with timer_cwc("cwc"):
            with timer_cwc("cwc/publishing_clustering"):
                clouds = [
                    c.cloud
                    .clone()
                    .transform(c.info.T_sensor2map)
                    .point.positions.numpy()
                    for c in clusters
                ]
                self.publish_pointclouds(
                    self._pub_debug_clusters,
                    clouds=clouds,
                    colors=[c.info.color for c in clusters],
                    frame_id=self._map_frame_id,
                )
            with timer_cwc("cwc/tree_manager"):
                path_odom_pos = np.array(
                    [
                        [p.pose.position.x, p.pose.position.y, p.pose.position.z]
                        for p in path_odom.poses
                    ]
                )
                path_odom_ori = [
                    Rotation.from_quat(
                        [
                            p.pose.orientation.x,
                            p.pose.orientation.y,
                            p.pose.orientation.z,
                            p.pose.orientation.w,
                        ]
                    )
                    for p in path_odom.poses
                ]
                pose_odom2map = self._tf_buffer.lookup_transform(
                    self._map_frame_id,
                    self._odom_frame_id,
                    Time(seconds=0),
                    Duration(seconds=1),
                )
                quat_odom2map = [
                    pose_odom2map.transform.rotation.x,
                    pose_odom2map.transform.rotation.y,
                    pose_odom2map.transform.rotation.z,
                    pose_odom2map.transform.rotation.w,
                ]
                R_odom2map = Rotation.from_quat(quat_odom2map)
                t_odom2map = np.array(
                    [
                        pose_odom2map.transform.translation.x,
                        pose_odom2map.transform.translation.y,
                        pose_odom2map.transform.translation.z,
                    ]
                )
                R_map2sensor = Rotation.from_matrix(
                    clusters[0].info.T_sensor2map[:3, :3].T
                )
                t_map2sensor = (
                    -clusters[0].info.T_sensor2map[:3, :3].T
                    @ clusters[0].info.T_sensor2map[:3, 3]
                )
                path_map = np.stack(
                    [
                        np.concatenate(
                            (
                                R_odom2map.apply(p) + t_odom2map,
                                (R_odom2map * o).as_quat(),
                            )
                        )
                        for p, o in zip(path_odom_pos, path_odom_ori)
                    ],
                    axis=0,
                )
                path_sensor = np.stack(
                    [
                        np.concatenate(
                            (
                                R_map2sensor.apply(p) + t_map2sensor,
                                (R_map2sensor * Rotation.from_quat(o)).as_quat(),
                            )
                        )
                        for p, o in zip(path_map[:, :3], path_map[:, 3:])
                    ]
                )

                verts_odom, tris = meshgrid_to_mesh(terrain)
                verts_map = R_odom2map.apply(verts_odom) + t_odom2map
                with self.tree_manager_lock:
                    self._tree_manager.add_clusters_with_path(clusters, path_sensor)
                    self._tree_manager.add_terrain(
                        verts_map,
                        tris,
                        clusters[0].info.time_stamp,
                        clusters[0].info.T_sensor2map,
                        self._terrain_cloth_cell_size,
                        None #path_map,
                    )

            with timer_cwc("cwc/publishing_tree_manager"):
                self.publish_tree_manager_state()
                # self._tree_manager.write_report(f"{self.base_output_path}/trees/logs")

            if self._debug_level > 1:
                with timer_cwc("cwc/dumping_clusters"):
                    cluster_dir = os.path.join(
                        self.base_output_path, "trees", str(self.last_pc_header.stamp)
                    )
                    if not os.path.exists(cluster_dir):
                        os.makedirs(cluster_dir, exist_ok=True)
                    else:
                        for f in os.listdir(cluster_dir):
                            os.remove(os.path.join(cluster_dir, f))
                    for cluster in clusters:
                        with open(
                            os.path.join(
                                cluster_dir,
                                f"tree{str(cluster.info.id).zfill(3)}.pkl",
                            ),
                            "wb",
                        ) as file:
                            pickle.dump(cluster, file)

        self.get_logger().info(
            f"Finished processing payload cloud {pc_counter}\n"\
            f"Timings of Cluster Worker (cw) and Cluster Worker Callback (cwc):\n{timer_cw}{timer_cwc}"
        )
        self._tree_manager.add_timing_result({"cw": timer_cw, "cwc": timer_cwc})
        gc.collect()
        return

    def path_callback(self, path_odom: Path):
        if len(path_odom.poses) < self.prev_path_len:
            if self.path_accumulators[0]["state"] == "active":
                self.path_accumulators[0]["state"] = "ready"
                assert self.path_accumulators[1]["state"] != "active", "Both path accumulators are active"
                self.path_accumulators[1]["state"] = "active"
            elif self.path_accumulators[1]["state"] == "active":
                self.path_accumulators[1]["state"] = "ready"
                assert self.path_accumulators[0]["state"] != "active", "Both path accumulators are active"
                self.path_accumulators[0]["state"] = "active"
        if self.path_accumulators[0]["state"] == "active":
            self.path_accumulators[0]["path"] = path_odom
        if self.path_accumulators[1]["state"] == "active":
            self.path_accumulators[1]["path"] = path_odom
        self.prev_path_len = len(path_odom.poses)

    def payload_callback(self, cloud_odom: PointCloud2):
        """This is the main callback function for the payload cloud and path. It
        initiates the clustering worker with the given parameters.

        Args:
            cloud_odom (PointCloud2): Cloud message in the odom frame
        """

        self.pc_counter += 1
        self.get_logger().info(f"Received payload cloud {self.pc_counter}")
        if len(cloud_odom.data) == 0:
            self.get_logger().warn("Aborting: Payload cloud empty!")
            return
        self.last_pc_header = cloud_odom.header
        if self.path_accumulators[0]["state"] == "ready":
            path_odom = self.path_accumulators[0]["path"]
        elif self.path_accumulators[1]["state"] == "ready":
            path_odom = self.path_accumulators[1]["path"]
        elif self.path_accumulators[0]["state"] == "active" and self.path_accumulators[1]["state"] == "empty":
            path_odom = self.path_accumulators[0]["path"] # use during accumulation for first path
        elif self.path_accumulators[1]["state"] == "active" and self.path_accumulators[0]["state"] == "empty":
            path_odom = self.path_accumulators[1]["path"] # use during accumulation for first path
        else:
            self.get_logger().error("No path available for payload cloud")
            return

        self._clustering_pool.apply_async(
            clustering_worker_fun,
            kwds={
                "tf_buffer": self._tf_buffer,
                "cloud_msg": cloud_odom,
                "terrain_enabled": self._terrain_enabled,
                "crop_radius": self._clustering_crop_radius,
                "path_odom": path_odom,
                "pose_graph_stamps": self.pose_graph_stamps,
                "crop_bounds": self._clustering_crop_bounds,
                "max_cluster_radius": self._clustering_max_cluster_radius,
                "precise_calcuation_fraction": self._clustering_distance_calc_point_fraction,
                "debug_level": self._debug_level,
                "map_frame_id": self._map_frame_id,
                "odom_frame_id": self._odom_frame_id,
                "normal_thr": self._clustering_normal_thr,
                "voxel_size": self._clustering_voxel_size,
            },
            callback=partial(
                self._clustering_worker_finished_callback,
                path_odom=path_odom,
                pc_counter=self.pc_counter,
            ),
        )

    def posegraph_changed_callback(self, posegraph_msg):
        """Callback for when the posegraph is updated. It updates the tree poses
        accordingly.

        Args:
            posegraph_msg: pose graph message
        """
        self.pose_graph_stamps = [
            pose.header.stamp for pose in posegraph_msg.poses
        ]
        pose_graph = [
            rospy_stamped_pose_to_stamped_pose(pose)
            for pose in posegraph_msg.poses
        ]

        if self.posegraph_updates_active:
            with self.tree_manager_lock:
                self._tree_manager.update_poses(pose_graph)

    def publish_tree_manager_state(self):
        """This function publishes the tree manager state, i.e. the trees, terrain,
        labels, canopie meshes, and point clouds
        """
        self.get_logger().info("Publishing tree manager state...")
        if len(self._tree_manager.trees) == 0:
            self.get_logger().error("No trees to be published!")
            return

        label_texts = []
        label_positions = []
        mesh_messages : MarkerArray = MarkerArray()
        canopy_messages : MarkerArray = MarkerArray()
        tree_manager_msg : TreeManagerMsg = TreeManagerMsg()

        # trees with labels
        tree_clouds = []
        tree_colors = []
        for tree, reco_flags, coverage_angle in zip(
            self._tree_manager.trees,
            self._tree_manager.tree_reco_flags,
            self._tree_manager.tree_coverage_angles,
        ):
            if np.max(tree.points[:, 2]) - np.min(tree.points[:, 2]) < 5.0:
                continue
            
            # append tree message to tree_manager_msg
            last_stamp = max([c.info.time_stamp for c in tree.clusters]).to_msg() # last stamp
            tree_msg = tree_to_tree_msg(
                tree,
                coverage_angle,
                self._map_frame_id,
                last_stamp
            )
            if tree_msg:
                tree_manager_msg.trees.append(tree_msg)

            if self.visibility_mask is not None and tree.id not in self.visibility_mask:
                continue

            if len(tree.clusters) < 3:
                continue

            # generate and publish label texts for RViz
            label_text = (
                f"##### tree{str(tree.id).zfill(3)} #####\n"
            )
            if tree.dbh:
                label_text += f"dbh:     {tree.dbh * 100:.1f} cm\n"
            else:
                label_text += f"dbh:    ({tree.axis.radius * 200:.1f}) cm\n"
            label_texts.append(label_text)
            label_positions.append(tree.axis.transform[:3, 3])

            terrain_verts, terrain_tris = tree.generate_mesh()
            mesh_messages.markers.append(
                generate_mesh_msg(
                    terrain_verts,
                    terrain_tris,
                    id=tree.id,
                    frame_id=self._map_frame_id,
                    time_stamp=self.last_pc_header.stamp,
                    color=[117 / 255, 49 / 255, 12 / 255],
                )
            )

            # sample random hue and for all clusters random brighntess between 0.5 and 1
            lightness = np.linspace(0.1, 0.9, len(tree.clusters))
            lightness = np.random.permutation(lightness)
            tree_colors.extend(
                # [[241, 148, 59] for _ in lightness]
                [colorsys.hls_to_rgb(tree.hue, l, 1.0) for l in lightness]
            )
            # add voxel downsampled pcs to tree_clouds
            tree.load_points()
            tree_clouds.extend(
                [
                    cluster.cloud
                    .clone()
                    .transform(cluster.info.T_sensor2map)
                    .point.positions.numpy()
                    for cluster in tree.clusters
                ]
            )
            tree.store_points()

            if tree.canopy_mesh is not None:
                message = generate_mesh_msg(
                    tree.canopy_mesh["vertices"],
                    tree.canopy_mesh["triangles"],
                    id=tree.id,
                    frame_id=self._map_frame_id,
                    time_stamp=self.last_pc_header.stamp,
                    color=[150 / 255, 217 / 255, 121 / 255],
                    alpha=0.2,
                )
                canopy_messages.markers.append(message)

        self._pub_stem_meshes.publish(mesh_messages)
        self._pub_canopy_meshes.publish(canopy_messages)
        self.publish_cluster_labels(label_texts, label_positions, self._map_frame_id)
        if len(tree_clouds) > 0:
            self.publish_pointclouds(
                self._pub_tree_clusters, tree_clouds, tree_colors, self._map_frame_id
            )

        # Terrain
        terrain_verts, terrain_tris = self._tree_manager.terrain
        self._pub_terrain_model.publish(
            generate_mesh_msg(
                terrain_verts,
                terrain_tris,
                frame_id=self._map_frame_id,
                time_stamp=self.last_pc_header.stamp,
                color=[191 / 255, 152 / 255, 124 / 255],
                alpha=1.0,
                id=0,
            )
        )
        tree_manager_msg.header.frame_id = self._map_frame_id
        tree_manager_msg.header.stamp = self.last_pc_header.stamp
        tree_manager_msg.terrain = mesh_to_mesh_msg(terrain_verts, terrain_tris)
        
        # Tree Manager 
        self._pub_tree_manager.publish(tree_manager_msg)

    def export_tree_manager(self):
        """exports the tree manager as a zip file to the output path"""
        stamp_now = self.get_clock().now()
        secs, nsecs = stamp_now.seconds_nanoseconds()
        path = os.path.join(
            self.base_output_path,
            "trees",
            "logs",
            "raw",
            f"tree_manager_{secs}_{nsecs:0>9}",
        )
        for tree in self._tree_manager.trees:
            tree.load_points()
        self._tree_manager.save_as_zip(path)

    def export_tree_manager_callback(self, request, response):
        self.export_tree_manager()
        return response

    def toggle_tree_visibility_callback(self, request, response):
        path = os.path.join(self.base_output_path, "tree_visibility_mask.txt")
        if not os.path.exists(path):
            self.get_logger().info("No tree visibility mask found under " + path)
            return response
        with open(path, "r") as file:
            visibility_mask = [int(line) for line in file.readlines() if line != ""]
        if visibility_mask == []:
            self.get_logger().info("No trees in mask, showing all trees")
            #return response
        if self.visibility_mask is None:
            self.visibility_mask = visibility_mask
            self.get_logger().info(
                "Now only showing trees with ids: " + str(visibility_mask)
            )
        else:
            self.visibility_mask = None
            self.get_logger().info("Now showing all trees")
        self.publish_tree_manager_state()
        return response

    def toggle_loop_closure_callback(self, request, response):
        self.posegraph_updates_active = not self.posegraph_updates_active
        self.get_logger().info(
            "Posegraph Updates now active"
            if self.posegraph_updates_active
            else "Posegraph Updates now inactive"
        )
        return response

    def on_shutdown(self):
        """
        Executes clears all stray files and memory before shutting down
        """
        for tree in self._tree_manager.trees:
            tree.remove_tmp_file()
        self._clustering_pool.close()
        return
