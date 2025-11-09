from copy import deepcopy
import os
import pickle
import shutil
from tempfile import NamedTemporaryFile
from typing import Any, List, Tuple
import zipfile

import numpy as np
import open3d as o3d
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.stats import multivariate_normal
import trimesh

from realtime_trees.tree import Tree
from realtime_trees.utils.ros_proxys import StampedPose, Time
from realtime_trees.utils.meshing import meshgrid_to_mesh
from realtime_trees.utils.distances import distance_line_to_line
from realtime_trees.utils.matrix_calc import apply_transform, efficient_inv
from realtime_trees.utils.gnss import map2lla
from realtime_trees.utils.dataclasses import Terrain, Cluster, CoverageInfo
from realtime_trees.utils.ros_proxys import StampedPose



class TreeManager:
    def __init__(
        self,
        distance_threshold: float = 0.5,
        reco_min_angle_coverage: float = 1.5 * np.pi,
        reco_min_distance: float = 4.0,
        terrain_confidence_stds: list = [3, 10, 10],
        terrain_confidence_sensor_weight: float = 0.9999,
        terrain_use_embree: bool = True,
        generate_canopy_mesh: bool = True,
        output_path: str = "/tmp",
        offload_to_disk: bool = False,
        debug_level: int = 0,
        payload_crop_radius: float = 20.0,
        reconstruction_args: dict = None,
        **kwargs,
    ) -> None:

        self.distance_threshold = distance_threshold
        self.reco_min_angle_coverage = reco_min_angle_coverage
        self.reco_min_distance = reco_min_distance
        self.terrain_confidence_stds = terrain_confidence_stds
        self.terrain_confidence_sensor_weight = terrain_confidence_sensor_weight
        self.use_embree = terrain_use_embree
        self.generate_canopy_mesh = generate_canopy_mesh
        self.base_output_path = output_path
        self.debug_level = debug_level
        self._offload_to_disk = offload_to_disk
        self.payload_crop_radius = payload_crop_radius
        if reconstruction_args is None:
            self.reconstruction_args = {
                "max_radius": np.inf,
                "max_consecutive_fails": 3,
            }
        else:
            self.reconstruction_args = reconstruction_args

        self.tree_reco_flags: List[List[bool]] = []
        self.tree_coverage_angles: List[float] = []
        self.trees: List[Tree] = []
        self._kd_tree: cKDTree = None

        self.num_trees = 0
        self._last_cluster_time = None

        self.terrains = []
        self.terrain_interpolator = None

        self.capture_Ts_with_stamps: List[StampedPose] = []
        self.timing_results = []

        self.lla_ref: np.ndarray = None
        self.lla_r2m = None

    @classmethod
    def from_zip(cls, path: str) -> "TreeManager":
        """Constructs a TreeManager from a Zip file as exported by
        TreeManager.save_as_zip

        Args:
            path (str): Path to the zip file

        Returns:
            TreeManager: Initialized Tree Manager
        """
        archive = zipfile.ZipFile(path, "r")
        tree_manager_dict: dict = pickle.loads(archive.read("tree_manager.pkl"))
        tree_manager = cls(**tree_manager_dict)
        tree_manager.tree_reco_flags = tree_manager_dict["tree_reco_flags"]
        tree_manager.tree_coverage_angles = tree_manager_dict["tree_coverage_angles"]
        tree_manager.num_trees = tree_manager_dict["num_trees"]
        tree_manager._last_cluster_time = tree_manager_dict["last_cluster_time"]
        tree_manager.capture_Ts_with_stamps = tree_manager_dict[
            "capture_Ts_with_stamps"
        ]
        tree_manager.timing_results = tree_manager_dict["timing_results"]
        tree_manager.terrains = tree_manager_dict["terrains"]
        tree_manager.reconstruction_args = (
            tree_manager_dict["reconstruction_args"]
            if "reconstruction_args" in tree_manager_dict
            else {}
        )
        tree_manager.lla_ref = tree_manager_dict.get("lla_ref", None)
        tree_manager.lla_r2m = tree_manager_dict.get("lla_r2m", None)

        for i in range(tree_manager.num_trees):
            try:
                tree_dict = pickle.loads(archive.read(f"tree_{i:0>5}/tree.pkl"))
            except KeyError:
                continue
            tree = Tree(**tree_dict)
            tree.reconstructed = tree_dict["reconstructed"]
            tree.circles = tree_dict["circles"]
            tree.canopy_mesh = tree_dict["canopy_mesh"]
            tree.clusters = tree_dict["clusters"]
            tree.dbh = tree_dict["dbh"]
            for j in range(len(tree.clusters)):
                with NamedTemporaryFile(mode="w+b", suffix=".pcd") as tmp_file:
                    tmp_file.write(
                        archive.read(
                            f"tree_{i:0>5}/cluster_{tree.clusters[j].info.time_stamp.secs}_{tree.clusters[j].info.time_stamp.nsecs:0>9}.pcd"
                        )
                    )
                    tmp_file.seek(0)
                    tree_cloud = o3d.io.read_point_cloud(tmp_file.name)
                    tree_cloud = o3d.t.geometry.PointCloud.from_legacy(tree_cloud)
                tree.clusters[j].cloud = tree_cloud
            tree_manager.trees.append(tree)

        tree_manager._update_kd_tree()  # initializes kd tree
        _ = tree_manager.terrain  # initializes terrain interpolator
        return tree_manager

    @property
    def stamps(self) -> np.ndarray:
        """Returns all time stamps of the clusters in the data base.

        Returns:
            np.ndarray: Sorted time stamps of all clusters in the data base.
        """
        if len(self.capture_Ts_with_stamps) == 0:
            return np.array([])
        return np.sort([Tws.time_stamp for Tws in self.capture_Ts_with_stamps])

    @property
    def terrain(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the terrain of the entire map in a 2.5D meshgrid format.

        Returns:
            np.ndarray, np.ndarray: Vertices and triangles of the terrain mesh.
        """

        # find most recent sensor transforms using time stamp of terrain maps
        if len(self.terrains) == 0 or len(self.capture_Ts_with_stamps) == 0:
            return None, None
        sensor_transforms = []
        terrains = []
        try:
            pose_graph_stamps = [t.time_stamp for t in self.capture_Ts_with_stamps]
        except KeyError:
            pose_graph_stamps = [t["stamp"] for t in self.capture_Ts_with_stamps]
        for terrain in self.terrains:
            if terrain.time_stamp not in pose_graph_stamps:
                print(f"terrain time stamp {terrain.time_stamp} not in posegraph")
                continue
            terrains.append(terrain)
            T_sensor2map = self.capture_Ts_with_stamps[
                pose_graph_stamps.index(terrain.time_stamp)
            ].pose
            sensor_transforms.append(T_sensor2map)

        # aggregate database entries into lists
        meshes_map = [
            deepcopy(terrain.mesh_sensor).apply_transform(terrain.T_sensor2map)
            for terrain in terrains
        ]

        bboxes = np.asarray(
            [
                (mesh.vertices.min(axis=0), mesh.vertices.max(axis=0))
                for mesh in meshes_map
            ]
        )

        # generate a regular grid of query points for the final terrain map
        bbox = [bboxes.reshape(-1, 3).min(axis=0), bboxes.reshape(-1, 3).max(axis=0)]
        (
            query_X,
            query_Y,
        ) = np.meshgrid(
            np.arange(bbox[0][0], bbox[1][0], terrains[0].cell_size),
            np.arange(bbox[0][1], bbox[1][1], terrains[0].cell_size),
        )
        query_positions = np.stack(
            (query_X, query_Y, -100 * np.ones_like(query_X)), axis=2
        )
        query_positions = query_positions.reshape(-1, 3)
        query_rays = np.zeros_like(query_positions)
        query_rays[:, 2] = 1.0

        # aggregate all maps into a single tensor of map heights and weights
        heights = np.ones((*query_X.shape, len(terrains)))
        weights = np.zeros((*query_X.shape, len(terrains)))
        for i, (terrain, mesh_map) in enumerate(zip(terrains, meshes_map)):
            mesh_weights = terrain.vertex_weights
            # querying all rays is not slower than preselecting the rays
            # inside the meshe's bounding box.
            # find indices in triangles where rays hit mesh
            tri_inds = mesh_map.ray.intersects_first(query_positions, query_rays)
            verts_mask = tri_inds != -1
            tri_inds = tri_inds[verts_mask]
            # just take the center of the tri as an approximation
            intersects = mesh_map.vertices[mesh_map.faces[tri_inds]].mean(axis=1)
            heights[verts_mask.reshape(query_X.shape), i] = intersects[:, 2]
            weights[verts_mask.reshape(query_X.shape), i] = mesh_weights[
                mesh_map.faces[tri_inds]
            ].mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            heights = np.sum(heights * weights, axis=2) / weights.sum(axis=2)

        # convert to vertices and triangles
        mgrid = np.stack((query_X, query_Y, heights), axis=2)
        self.terrain_interpolator = RegularGridInterpolator(
            points=(query_X[0, :], query_Y[:, 0]),
            values=heights.T,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        verts, tris = meshgrid_to_mesh(mgrid)

        # remove verts with nan
        nan_mask = np.isnan(verts[:, 2])
        # remove verts where there are no maps contributing strongly
        weights_mask = weights.max(axis=2) < 0.2 * (
            1 - self.terrain_confidence_sensor_weight
        )

        # # for debugging puproposes, use this to tune the value of the threshold
        # from matplotlib import pyplot as plt
        # # plt.imshow(weights_mask.reshape(query_X.shape))
        # plt.imshow(weights_mask.reshape(query_X.shape)*1 + nan_mask.reshape(query_X.shape)*1)
        # # plt.imshow(weights.max(axis=2))
        # plt.colorbar()
        # plt.show()

        verts_mask = np.logical_or(nan_mask, weights_mask.reshape(-1))
        remove_indices = np.where(verts_mask)[0]
        tri_mask = np.any(np.isin(tris, remove_indices), axis=1)
        tris = tris[~tri_mask]

        # flip normals
        tris = np.flip(tris, axis=1)

        return verts, tris

    def register_gnss(self, lla_ref: np.ndarray, lla_r2m: np.ndarray):
        """Rrgisters the map frame where all map measures are represented in, to the LLA
        frame given references obtained by a GNSS system.

        Args:
            lla_ref (np.ndarray): reference point on the earth in LLA coordinated
            lla_r2m (np.ndarray): transfrom from reference to map frame
        """
        self.lla_ref = lla_ref
        self.lla_r2m = lla_r2m

    def _update_kd_tree(self) -> None:
        """Updates the KD tree with the current tree centers"""
        centers = [tree.axis.transform[:2, 3] for tree in self.trees]
        if len(centers) == 0:
            return
        self._kd_tree = cKDTree(centers)

    def _new_tree_from_cluster(self, cluster: Cluster) -> None:
        """adds a new tree given a single cluster.

        Args:
            cluster (Cluster): Cluster object as in the list returned by
            realtime_trees.clustering.cluster
        """
        new_tree = Tree(
            id=self.num_trees,
            place_holder_height=cluster.info.axis.height,
            tmp_path=self.base_output_path if self._offload_to_disk else None,
            payload_crop_radius=self.payload_crop_radius,
        )
        new_tree.add_cluster(cluster)
        self.trees.append(new_tree)
        self.tree_reco_flags.append({"angle_flag": False, "distance_flag": False})
        self.tree_coverage_angles.append(0.0)
        self.num_trees += 1

    def add_clusters(
        self, clusters_sensor: List[Cluster], try_reconstruction: bool = True
    ) -> None:
        """Checks every cluster (in SENSOR FRAME). If a tree close to the
        detected cluster already exists, the cluster is added to the tree. If no tree is
        close enough, a new tree is created. The KD tree is updated as well.

        Args:
            clusters (List[Cluster]): List of clusters as returned by
                realtime_trees.clustering.cluster
            try_reconstruction (bool): If true, reconstruction is attempted after adding
                all clusters. Defaults to True.
        """
        if len(clusters_sensor) == 0:
            return
        # time stamp and sensor transform are same for all clusters, so just take first
        self.capture_Ts_with_stamps.append(
            StampedPose(
                time_stamp=clusters_sensor[0].info.time_stamp,
                pose=clusters_sensor[0].info.T_sensor2map
                
            )
        )

        self._last_cluster_time = clusters_sensor[0].info.time_stamp
        candidate_transforms_odom = [
            c.info.T_sensor2map @ c.info.axis.transform
            for c in clusters_sensor
        ]
        candidate_centers = [t[:2, 3] for t in candidate_transforms_odom]

        if len(self.trees) == 0:
            # create new trees for all clusters and add them to the list
            for cluster in clusters_sensor:
                self._new_tree_from_cluster(cluster)
        else:
            # for all clusters check if tree at this coordinate already exists
            num_existing, num_new = 0, 0
            _, existing_indices = self._kd_tree.query(candidate_centers)
            for i_candidate, i_existing in enumerate(existing_indices):
                candidate_axis = deepcopy(clusters_sensor[i_candidate].info.axis)
                # transform candidate axis to odom frame
                candidate_axis.transform = candidate_transforms_odom[i_candidate]
                existing_axis = self.trees[i_existing].axis

                distance = distance_line_to_line(
                    candidate_axis, existing_axis, clip_heights=[0, 10]
                )
                if distance < self.distance_threshold:
                    self.trees[i_existing].add_cluster(clusters_sensor[i_candidate])
                    num_existing += 1
                else:
                    self._new_tree_from_cluster(clusters_sensor[i_candidate])
                    num_new += 1

            print(f"Found {num_existing} existing and {num_new} new clusters")

        self._update_kd_tree()
        if try_reconstruction:
            self.try_reconstructions()

    def add_clusters_with_path(
        self,
        clusters_sensor: List[Cluster],
        path_sensor: np.ndarray,
        try_reconstruction: bool = True,
    ) -> None:
        """This function checks every cluster and performs the same as add_clusters().
        In addition, it calculates the covered angle and covered distance of the sensor
        to the tree axis for every cluster and adds this information to the
        cluster.info.

        Args:
            clusters (List[Cluster]): List of clusters as returned by
                realtime_trees.clustering.cluster
            path (np.ndarray): Nx3 array describing consecutive 3D poses of the sensor
                in SENSOR FRAME.
                The columns are the x, y, z coordinates of the position.
            try_reconstruction (bool): If true, reconstruction is attempted after adding
                all clusters. Defaults to True.
        """

        for cluster_sensor in clusters_sensor:
            angle_from, angle_to, d_min, d_max = self.calculate_coverage(
                cluster_sensor, path_sensor
            )
            cluster_sensor.info.coverage = CoverageInfo(
                angle_from=angle_from,
                angle_to=angle_to,
                distance_min=d_min,
                distance_max=d_max
            )

        self.add_clusters(clusters_sensor, try_reconstruction)

    def add_timing_result(self, timer: dict) -> None:
        """adds a timing result to the tree manager for logging purposes.

        Args:
            timer (dict): Dict containing realtime_trees.utils.timing.Timer objects.
        """
        self.timing_results.append(timer)

    def add_terrain(
        self,
        terrain_mesh_verts_map: np.ndarray,
        terrain_mesh_tris: np.ndarray,
        time_stamp: Time,
        T_sensor2map: np.ndarray,
        cell_size: float,
        path_map: np.ndarray = None,
    ) -> None:
        """adds a single terrain from a payload to the tree manager. The terrain is
        represented in the MAP frame.

        Args:
            terrain_mesh_verts_map (np.ndarray): Vertices of the local terrain in MAP
                frame.
            terrain_mesh_tris (np.ndarray): Triangles describing the mesh topology of
                the terrain.
            time_stamp (Time): time stamp of the terrain for SLAM association.
            T_sensor2map (np.ndarray): transform from the sensor to the map frame to
                associate the local terrain to the SLAM pose graph.
            cell_size (float): cell size of the terrain to be added.
            path_map (np.ndarray): path of the sensor while capturing the terrain in
                the MAP frame. Defaults to None.
        """

        def measurement_likelihood(verts: np.ndarray, path: np.ndarray = None):
            """Calculates a measurement likelihood by weighting the area behind the
            sensor with higher weights. This is used to allow for backwards-angled
            sensors.
            This likelihood is used to weight the terrains for global aggregation. If no
            path is provided, only the C1 weights are used.

            Args:
                verts (np.ndarray): Nx3 Array of vertics
                path (np.ndarray): Mx7 Array of sensor poses (x, y, z, qx, qy, qz, qw).
                    Defaults to None.

            Returns:
                np.ndarray: Nx1 Array of weights
            """
            if path is not None:
                # calculate distances from sensor to all verts
                rot_mats = np.array(
                    [Rotation.from_quat(p[3:]).as_matrix() for p in path]
                )
                means = 9.0 * rot_mats[:, :, 0] + path[:, :3]
                variances = (
                    rot_mats
                    @ np.diag(self.terrain_confidence_stds) ** 2
                    @ rot_mats.transpose([0, 2, 1])
                )
                weights_sensor = [
                    multivariate_normal.pdf(verts, mean, var)
                    for mean, var in zip(means, variances)
                ]
                weights_sensor = np.array(weights_sensor).sum(axis=0)
                weights_sensor /= weights_sensor.max()
                weights_distance = np.linalg.norm(
                    verts[:, None, :] - path[:, :3], axis=2
                ).min(axis=1)
                max_distance = weights_distance.max()
                weights_distance = np.sqrt(0.5) * max_distance - weights_distance
                weights_distance /= np.sqrt(0.5) * max_distance
                weights_distance = np.clip(weights_distance, 0, 1)
                weights = (
                    (1 - self.terrain_confidence_sensor_weight) * weights_distance
                    + self.terrain_confidence_sensor_weight * weights_sensor
                )
            else:
                center_vertex = np.mean(verts, axis=0)
                dists = np.linalg.norm(verts - center_vertex, axis=1)
                max_distance = dists.max()
                weights = np.sqrt(0.5) * max_distance - dists
                weights /= np.sqrt(0.5) * max_distance
                weights = np.clip(weights, 0, 1)

            return weights

        weights = measurement_likelihood(terrain_mesh_verts_map, path_map)
        terrain_mesh_sensor_verts = apply_transform(
            terrain_mesh_verts_map,
            T_sensor2map[:3, 3],
            T_sensor2map[:3, :3],
            inverse=True,
        )

        terrain_mesh_sensor = trimesh.Trimesh(
            terrain_mesh_sensor_verts.astype(np.float64),
            terrain_mesh_tris.astype(np.int64),
            use_embree=self.use_embree,
        )

        self.terrains.append(
            Terrain(
                mesh_sensor=terrain_mesh_sensor,
                time_stamp=time_stamp,
                T_sensor2map=T_sensor2map,
                vertex_weights=weights,
                path=path_map,
                cell_size=cell_size
            )
        )

    def update_poses(self, new_posegraph: List[StampedPose]) -> None:
        """detects changes in the posegraph and updates the coordinate systems of all
        clusters in all trees.

        Args:
            posegraph_poses (List[StampedPose]): List of of all poses of the posegraph
                where the timestamp is used as the pose's unique ID.
        """
        # find all poses that have changed since capture
        new_stamps = [pose.time_stamp for pose in new_posegraph]
        changed_poses = []
        for i, T_with_stamp in enumerate(self.capture_Ts_with_stamps):
            try:
                i_new = new_stamps.index(T_with_stamp.time_stamp)
                new_T_map = new_posegraph[i_new].pose

                if not np.allclose(new_T_map, T_with_stamp.pose):
                    changed_poses.append(
                        StampedPose(
                            time_stamp=T_with_stamp.time_stamp,
                            pose=new_T_map
                        )
                    )
                    self.capture_Ts_with_stamps[i].pose = new_T_map
            except ValueError:
                print("Timestamp not found")
                print(self.capture_Ts_with_stamps)
                continue

        # realign clusters and adjust meshes accordingly
        for tree in self.trees:
            delta_Ts = []
            for cluster in tree.clusters:
                pose_changed = False
                for changed_pose in changed_poses:
                    if changed_pose.time_stamp == cluster.info.time_stamp:
                        pose_changed = True
                        delta_Ts.append(
                            changed_pose.pose
                            @ efficient_inv(cluster.info.T_sensor2map)
                        )
                        cluster.info.T_sensor2map = changed_pose.pose
                        break
                if not pose_changed:
                    delta_Ts.append(np.eye(4))
            mean_t = np.mean([T[:3, 3] for T in delta_Ts], axis=0)
            mean_rot = (
                Rotation.from_matrix(np.stack([T[:3, :3] for T in delta_Ts], axis=0))
                .mean()
                .as_matrix()
            )
            tree.transform_circles(mean_t, mean_rot)
        # realign terrains
        for terrain in self.terrains:
            if terrain.time_stamp in new_stamps:
                terrain.T_sensor2map = new_posegraph[
                    new_stamps.index(terrain.time_stamp)
                ].pose
        self._update_kd_tree()
        self.try_remerge_trees()

    def try_remerge_trees(self) -> None:
        """Analyzes the tree database and merges trees that are now close enough to
        each other (due to posegraph updates)
        """
        if self._kd_tree is None:
            return
        while True:
            merges = self._kd_tree.query_ball_tree(
                self._kd_tree, r=self.distance_threshold
            )
            merges = sorted(
                list(set([tuple(sorted(r)) for r in merges if len(r) > 1])),
                key=lambda x: len(x),
                reverse=True,
            )
            if len(merges) == 0:
                break
            merge: List[int] = merges[0]
            remaining_tree = self.trees[merge[0]]
            remaining_tree.merge([self.trees[i] for i in merge[1:]])
            if np.any([self.trees[i].reconstructed for i in merge]):
                self.analyze_tree(remaining_tree, reconstruct=True)
            for i in merge[1:]:
                self.trees.pop(i)
            self._update_kd_tree()
            print(f"Merged trees {merge}")

    def calculate_coverage(
        self,
        cluster_sensor: Cluster,
        path_sensor: np.ndarray,
    ) -> Tuple[float]:
        """Calculates the covered angle of the sensor to the tree axis for a given
        cluster. The covered angle is given by two values defining the global extent
        of the arc around the z-axis in the x-y-plane bounding all rays from the
        center of the tree axis to all sensor poses. The parametrization is given by a
        start angle and end angle.
        The angles are given wrt. to the global x-axis and are in [0, 2*pi].

        Args:
            cluster_sensor (Cluster): Cluster object representing the sensor cluster.
            path_odom (np.ndarray): Nx7 array describing consecutive 7D poses of the
                sensor in the ODOM FRAME.
                The first three columns are the x, y, z coordinates of the position. The
                last four columns are the x, y, z, w quaternions describing orientation.

        Returns:
            float: start angle of arc wrt to global x-axis
            float: end angle of arc wrt to global x-axis
            float: min distance from sensor pose to tree axis
            float: max distance from sensor pose to tree axis
        """
        min_distance = np.inf
        max_distance = -np.inf
        angles = []
        for pose in path_sensor:
            # calculate connecting vector between sensor pose and tree axis center
            ray_vector = pose[:2] - cluster_sensor.info.axis.transform[:2, 3]
            ray_length = np.linalg.norm(ray_vector)

            # calculate angle of ray vector wrt. global x-axis
            ray_vector /= ray_length
            angle = np.arctan2(-ray_vector[1], -ray_vector[0]) + np.pi  # range 0 to 2pi
            angles.append(angle)

            # update min and max distance and angle
            min_distance = min(min_distance, ray_length)
            max_distance = max(max_distance, ray_length)

        angles = np.unwrap(np.asarray(angles))

        coverage = np.abs(angles.max() - angles.min())
        if np.isclose(coverage, 2 * np.pi) or coverage > 2 * np.pi:
            angle_from, angle_to = 0, 2 * np.pi
        else:
            angle_from, angle_to = angles.min() % (2 * np.pi), angles.max() % (
                2 * np.pi
            )

        return angle_from, angle_to, min_distance, max_distance

    def compute_angle_coverage(self, intervals: List[Tuple[float]]) -> float:
        """Calculates the coverd angle of the union of the given angle intervals.

        Args:
            intervals (List[Tuple[float]]): Tuple of angles. Each tuple contains two
                angles defining the start and end angle of an arc wrt. the global
                x-axis.


        Returns:
            float: coverage angle in rad
        """
        angle_accumulator = np.zeros(360, dtype=bool)
        for angle_from, angle_to in intervals:
            angle_from = int(np.around(np.rad2deg(angle_from)))
            angle_to = int(np.around(np.rad2deg(angle_to)))
            if angle_to < angle_from:
                angle_accumulator[angle_from:] = True
                angle_accumulator[:angle_to] = True
            else:
                angle_accumulator[angle_from:angle_to] = True
        return np.deg2rad(angle_accumulator.sum())

    def try_reconstructions(self) -> bool:
        """Checks all trees in the data base if they satisfy the conditions to be
        reconstructed. If so, they are reconstrcuted.

        Returns:
            bool: True if at least one tree was newly reconstructed, False otherwise.
        """
        reco_happened = False
        for i, tree in enumerate(self.trees):
            coverages = [
                (
                    cluster.info.coverage.angle_from,
                    cluster.info.coverage.angle_to,
                    cluster.info.coverage.distance_min,
                    cluster.info.coverage.distance_max,
                )
                for cluster in tree.clusters
            ]

            coverage_angle = self.compute_angle_coverage([c[:2] for c in coverages])
            self.tree_coverage_angles[tree.id] = coverage_angle
            angle_flag = coverage_angle > self.reco_min_angle_coverage
            self.tree_reco_flags[i]["angle_flag"] = angle_flag

            distances = np.array([c[2:] for c in coverages])
            d_min = np.min(distances[:, 0])
            distance_flag = d_min < self.reco_min_distance
            self.tree_reco_flags[i]["distance_flag"] = distance_flag

            if (
                len(tree.clusters) == tree.num_clusters_after_last_reco
                and not tree.cosys_changed_after_last_reco
            ):
                continue

            if not angle_flag or not distance_flag:
                continue

            if self.debug_level > 0:
                print(f"Reconstructing tree {tree.id}")

            reco_sucess = self.analyze_tree(tree, reconstruct=True)
            reco_happened |= reco_sucess

        return reco_happened

    def analyze_tree(self, tree: Tree, reconstruct: bool = False) -> bool:
        """Analyzes the tree by reconstructing it and updating its tree traits

        Args:
            tree (Tree): Tree to be analyzed

        Returns:
            bool: Flag if reconstruction has been successful
        """
        if reconstruct:
            tree.reconstruct(**self.reconstruction_args)
        if tree.reconstructed:
            tree.num_clusters_after_last_reco = len(tree.clusters)
            tree.cosys_changed_after_last_reco = False
            if self.terrain_interpolator is None:  # initialize terrain interpolator
                _ = self.terrain
            if self.generate_canopy_mesh:
                tree.generate_canopy()
            if self.terrain_interpolator:
                terrain_height = self.terrain_interpolator(
                    tree.axis.transform[:2, 3]
                )[0]
                if not np.isnan(terrain_height):
                    tree.compute_dbh(terrain_height)
                return True
        return False

    def write_report(self, path: str) -> None:
        """Writes the tree data base to a csv file

        Args: path (str, optional): Path to the directory where the csv and xlsx file is
            saved.
        """
        if self._last_cluster_time is None:
            return
        file_name = (
            "TreeManagerState_"
            + f"{self._last_cluster_time.secs}_{self._last_cluster_time.nsecs:0>9}"
        )
        file_name_csv = file_name + ".csv"
        file_name_xlsx = file_name + ".xlsx"

        # write header
        columns = [
            "tree_id",
            "location_x",
            "location_y",
            "number_clusters",
            "coverage_angle",
            "reconstructed",
            "dbh",
            "dbh_approximation",
        ]
        if self.lla_ref is not None and self.lla_r2m is not None:
            columns.insert(3, "location_lat")
            columns.insert(4, "location_long")
        # write tree data
        data = []
        for tree in self.trees:
            if len(tree.clusters) < 3:
                continue
            # don't export bushes
            if tree.get_height(self.terrain_interpolator(tree.axis.transform[:2,3])) < 3:
                continue
            entry = [
                tree.id,
                tree.axis.transform[0, 3],
                tree.axis.transform[1, 3],
                len(tree.clusters),
                np.rad2deg(self.tree_coverage_angles[tree.id]),
                tree.reconstructed,
                tree.dbh,
                tree.axis.radius * 2,
            ]
            if self.lla_ref is not None and self.lla_r2m is not None:
                lla_coords = map2lla(tree.axis.transform[:3, 3], self.lla_ref, self.lla_r2m)
                entry.insert(3, lla_coords[0])
                entry.insert(4, lla_coords[1])
            data.append(entry)
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(os.path.join(path, file_name_csv), float_format="%.8f", index=False)
        df.to_excel(os.path.join(path, file_name_xlsx), index=False)

    def get_payload_map(self, stamp: Time) -> np.ndarray:
        """Generates a map of all detections in a single payload. This map is 2D and
        only contains tree position and its very rough radius.

        Args:
            stamp (Time): time stamp of the payload

        Returns:
            np.ndarray: Nx3 array with x, y, radius
        """
        try:
            map = np.concatenate(
                [
                    np.hstack(
                        (
                            (
                                cluster.info.T_sensor2map
                                @ cluster.info.axis.transform
                            )[:2, 3],
                            np.array(cluster.info.axis.radius),
                        )
                    )[None, :]
                    for tree in self.trees
                    for cluster in tree.clusters
                    if cluster.info.time_stamp == stamp
                ],
                axis=0,
            )
        except ValueError:
            return np.empty((1, 3))
        return map

    def get_full_map(self) -> np.ndarray:
        """Generates a 2D array with a tree object for every tree. Each row contains the
        location and tree traits of a tree.

        Returns:
            np.ndarray: 2D array of all trees.
        """
        map = np.vstack(
            [
                np.hstack((tree.axis.transform[:2, 3], tree.axis.radius))
                for tree in self.trees
            ]
        )
        return map

    def save_as_zip(self, path: str, hq_only: bool = True) -> None:
        """Exports the tree manager to a zip file. The zip file contains all trees and
        all terrain maps. The tree manager is saved as a pickle file.

        Args:
            path (str): Filename to save the Zip
            hq_only (bool): Only exports trees with good reconstructions to the excel 
                sheet. Still, the complete state of the tree manager is exported.
        """
        if path.endswith(".zip"):
            path = path.replace(".zip", "")
        print(f"Saving to zipfile {path}.zip")
        try:
            os.makedirs(path, exist_ok=True)
            for tree in self.trees:
                tree.write_to_disk(path)
                if tree.reconstructed:
                    # if reconstruction is available, export the full pc for further processing
                    tree_cloud = o3d.t.geometry.PointCloud(tree.points).to_legacy()
                    o3d.io.write_point_cloud(
                        os.path.join(
                            path,
                            f"tree_{tree.id:0>3}.pcd",
                        ),
                        tree_cloud,
                    )

            export_dict = {
                "distance_threshold": self.distance_threshold,
                "reco_min_angle_coverage": self.reco_min_angle_coverage,
                "reco_min_distance": self.reco_min_distance,
                "terrain_confidence_stds": self.terrain_confidence_stds,
                "terrain_confidence_sensor_weight": self.terrain_confidence_sensor_weight,
                "terrain_use_embree": self.use_embree,
                "generate_canopy_mesh": self.generate_canopy_mesh,
                "output_path": self.base_output_path,
                "debug_level": self.debug_level,
                "offload_to_disk": self._offload_to_disk,
                "payload_crop_radius": self.payload_crop_radius,
                "tree_reco_flags": self.tree_reco_flags,
                "tree_coverage_angles": self.tree_coverage_angles,
                "num_trees": self.num_trees,
                "last_cluster_time": self._last_cluster_time,
                "capture_Ts_with_stamps": self.capture_Ts_with_stamps,
                "timing_results": self.timing_results,
                "terrains": self.terrains,
                "reconstruction_args": self.reconstruction_args,
                "lla_ref": self.lla_ref,
                "lla_r2m": self.lla_r2m,
            }

            with open(os.path.join(path, "tree_manager.pkl"), "wb") as file:
                pickle.dump(export_dict, file)
            self.write_report(path)
            # export terrain as obj
            # trimesh.Trimesh(*self.terrain).export(os.path.join(path, "terrain.obj"))

            shutil.make_archive(path, "zip", path)
        finally:
            shutil.rmtree(path)
            print("finished export")
