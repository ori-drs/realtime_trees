import gc
import os
from copy import deepcopy
import pickle
from typing import Iterable, List, Union, Tuple

from matplotlib import axis
from networkx import radius
import numpy as np
import open3d as o3d
from scipy import cluster
from scipy.spatial.transform import Rotation
import trimesh

from realtime_trees.utils.distances import pnts_to_axes_sq_dist
from realtime_trees.circle import Circle
from realtime_trees.utils.dataclasses import Cluster, ClusterAxis


class Tree:
    def __init__(
        self,
        id: int,
        place_holder_height: float = 5,
        tmp_path: str = None,
        **kwargs,
    ) -> None:
        """Constructs the Tree object.

        Args:
            id (int): Unique identifier of tree
            place_holder_height (float, optional): Height of placeholder when it is
                generated. Defaults to 5.
            tmp_path (str, optional): Temporary path to save this tree's clusters to
                disk. If None, this feature cannot be used. Defaults to None.
        """
        self.id = id
        self.place_holder_height = place_holder_height
        self._tmp_path = tmp_path

        self.reconstructed = False
        self.circles: List[Circle] = None
        self.canopy_mesh = None

        self.clusters = []

        self.num_clusters_after_last_reco = 0
        self.cosys_changed_after_last_reco = False

        self.hue = np.random.rand()
        self.dbh = None
        

    def merge(self, other_trees: List["Tree"]):
        """Merges self with other trees. This resets all reconstructions! So make sure
        to re-reconstruct the tree and re-calculate tree traits.

        Args:
            other_trees (List[Tree]): List of other trees to merge
        """
        for other_tree in other_trees:
            self.clusters.extend(other_tree.clusters)
        self.reset()

    def reset(self):
        """Resets all tree to initialization state."""
        self.reconstructed = False
        self.circles = None
        self.canopy_mesh = None
        self.num_clusters_after_last_reco = 0
        self.cosys_changed_after_last_reco = False

    def add_cluster(self, cluster: Cluster):
        """Adds a cluster to the tree. This resets all reconstructions! So make sure

        Args:
            cluster (Cluster): Cluster object as returned by
            realtime_trees.clustering.cluster
        """
        self.load_points()
        self.clusters.append(cluster)
        self.store_points()

    def load_points(self):
        """Loads all points that are stored to temporary pcd files from disk."""
        if self._tmp_path is None:
            return
        path = os.path.join(self._tmp_path, "tree_points", f"tree_{self.id:0>5}.pkl")
        if not os.path.exists(path):
            return
        with open(path, "rb") as file:
            points = pickle.load(file)
        assert len(points) == len(
            self.clusters
        ), "Something went wrong with loading the points from disk"

        for i in range(len(self.clusters)):
            self.clusters[i].cloud = points[i]

    def store_points(self):
        """Stores all points to temporary pcd files on disk. This can reduce memory use
        as points are only loaded into RAM when needed.
        CAVEAT: This works not reliably with the python garbage collector not really
        behaving as predicted...
        """
        if self._tmp_path is None:
            return
        path = os.path.join(self._tmp_path, "tree_points")
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, f"tree_{self.id:0>5}.pkl")
        with open(filename, "wb") as file:
            pickle.dump([c.cloud for c in self.clusters], file)
        for i in range(len(self.clusters)):
            del self.clusters[i].cloud
        gc.collect()  # actually delete the points

    @property
    def axis(self) -> ClusterAxis:
        """Returns Axis of the tree in the MAP FRAME. The axis is calculated as the
        mean of all axis parameters of the clusters.

        Raises:
            ValueError: If there are no clusters

        Returns:
            ClusterAxis: ClusterAxis object representing themean axis
        """
        if len(self.clusters) == 0:
            raise ValueError("No measurements available yet.")

        cluster_trafos_odom = [
            c.info.T_sensor2map @ c.info.axis.transform
            for c in self.clusters
        ]

        mean_center = np.mean([c[:3, 3] for c in cluster_trafos_odom], axis=0)
        mean_radius = np.mean(
            [c.info.axis.radius for c in self.clusters], axis=0
        )

        # mean rotation is a bit trickier, scipy takes care of that
        rotation_stack = Rotation.from_matrix([c[:3, :3] for c in cluster_trafos_odom])
        mean_rotation = rotation_stack.mean().as_matrix()

        mean_T = np.eye(4)
        mean_T[:3, :3] = mean_rotation
        mean_T[:3, 3] = mean_center

        if self.reconstructed:
            height = self.circles[-1].center[2] - self.circles[0].center[2]
        else:    
            height = self.place_holder_height
        
        return ClusterAxis(
            transform=mean_T,
            radius=mean_radius,
            height=height,
            score=0,
        )

    @property
    def points(self) -> np.ndarray:
        """Returns all the tree's points in the MAP frame

        Raises:
            ValueError: if there are no points in the tree.

        Returns:
            np.ndarray: numpy array of shape (n, 3) with all points
        """
        if len(self.clusters) == 0:
            raise ValueError("No measurements available yet.")

        self.load_points()
        # transform all points to odom frame and then stack them
        points = np.vstack(
            [
                cluster.cloud.point.positions.numpy()
                @ cluster.info.T_sensor2map[:3, :3].T
                + cluster.info.T_sensor2map[:3, 3]
                for cluster in self.clusters
            ]
        )
        self.store_points()
        return points

    @property
    def canopy_volume(self) -> float:
        """Computes the volume of the canopy mesh (a tree trait)

        Returns:
            float: Canopy volume in m^3
        """
        if self.canopy_mesh is not None:
            canopy_mesh = trimesh.Trimesh(
                vertices=self.canopy_mesh["vertices"], faces=self.canopy_mesh["triangles"]
            )
            return canopy_mesh.volume
        else:
            return None

    def get_height(self, terrain_height) -> float:
        return np.percentile(self.points[:, 2], 99) - terrain_height

    def transform_circles(self, translation: np.ndarray, rotation: np.ndarray):
        """applies the transform to all member objects of this tree.

        Args:
            translation (np.ndarray): 3x1 translation vector
            rotation (np.ndarray): Either 3x3 rotation matrix or 4x1 quaternion

        Raises:
            ValueError: if rotation is not given as 3x3 matrix or 4x1 quaternion
        """
        if self.reconstructed:
            for i in range(len(self.circles)):
                self.circles[i].apply_transform(translation, rotation)

    def apply_transform(self, translation: np.ndarray, rotation: np.ndarray):
        """applies the transform to all member objects of this tree.

        Args:
            translation (np.ndarray): 3x1 translation vector
            rotation (np.ndarray): Either 3x3 rotation matrix or 4x1 quaternion

        Raises:
            ValueError: if rotation is not given as 3x3 matrix or 4x1 quaternion
        """
        self.transform_circles(translation, rotation)
        T = np.eye(4)
        if rotation.shape[0] == 4:
            T[:3, :3] = Rotation.from_quat(rotation).as_matrix()
        elif rotation.shape == (3, 3):
            T[:3, :3] = rotation
        else:
            raise ValueError("rotation must be given as 3x3 matrix or quaternion")
        T[:3, 3] = translation
        for i in range(len(self.clusters)):
            self.clusters[i].info.axis.transform = (
                T @ self.clusters[i].info.axis.transform
            )

    def evaluate_at_height(self, height: float) -> Circle:
        """Evaluates the tree curve center and the radius at a given height. If the tree

        Args:
            height (float): Height for evaluation

        Returns:
            Circle: curve center and radius represented as a Circle object
        """
        circ_interp = None
        if not self.reconstructed:
            return circ_interp

        # extrapolate up to one slice interval under first slice
        first_frustum_height = self.circles[1].center[2] - self.circles[0].center[2]
        if -first_frustum_height < height - self.circles[0].center[2] < 0:
            alpha = (self.circles[0].center[2] - height) / first_frustum_height
            center = self.circles[0].center + alpha * (
                self.circles[0].center - self.circles[1].center
            )
            radius = self.circles[0].radius + alpha * (
                self.circles[0].radius - self.circles[1].radius
            )
            if np.any(np.isnan(center)) or np.isnan(radius):
                return None
            return Circle(center, radius)

        # interpolate elsewhere
        for i_circ in range(len(self.circles) - 1):
            if (
                height < self.circles[i_circ + 1].center[2]
                and height > self.circles[i_circ].center[2]
            ):
                alpha = (height - self.circles[i_circ].center[2]) / (
                    self.circles[i_circ + 1].center[2] - self.circles[i_circ].center[2]
                )
                center = (1 - alpha) * self.circles[
                    i_circ
                ].center + alpha * self.circles[i_circ + 1].center
                radius = (1 - alpha) * self.circles[
                    i_circ
                ].radius + alpha * self.circles[i_circ + 1].radius
                if np.any(np.isnan(center)) or np.isnan(radius):
                    return None
                return Circle(center, radius)

    def _filter_slice(
        self,
        sliced_clusters: List[np.ndarray],
        radius_lb: float,
        radius_ub: float,
        center_region: Circle,
        filter_radius: float,
        slice_height: float,
    ) -> dict:
        """Filters a set of sliced clusters by fitting circles to them and filtering
        points by distance to the fitted circle

        Args:
            sliced_clusters (List[np.ndarray]): list of sliced clusters
            radius_lb (float): lower bound for radius of filter circle
            radius_ub (float): upper bound for radius of filter circle
            center_region (Circle): initial guess for a center region. Providing this,
                significantly reduces compute time.
            filter_radius (float): distance for a point to be considered inlier to the
                circle fit
            slice_height (float): height of the slice. This is only used to initialize
                the fitted circle's height.

        Returns:
            dict : Dict containing the filter circles, scores, point counts, filtered
                points and filter indices. If no fit is found, None is returned.
        """
        hough_ransac_circles = []
        scores = []
        point_counts = []
        remove_inds = []
        filter_indices = []
        for i_cluster, points in enumerate(sliced_clusters):
            # too few points
            if len(points) < 10:
                remove_inds.append(i_cluster)
                continue

            # fit circle
            filter_circle = Circle.from_cloud_hough_ransac(
                points,
                min_radius=radius_lb,
                max_radius=radius_ub,
                center_region=center_region,
                circle_height=slice_height,
            )
            # # alternative fitting of filter circle using hough algorithm
            # filter_circle = Circle.from_cloud_hough(
            #     points,
            #     min_radius=radius_lb,
            #     max_radius=radius_ub,
            #     center_region=center_region,
            #     circle_height=slice_height,
            # )
            # # alternative fitting of filter circle using ransac algorithm
            # filter_circle = Circle.from_cloud_ransac(
            #     points,
            #     circle_height=slice_height,
            #     circle_height=slice_height,
            #     min_radius=radius_lb,
            #     max_radius=radius_ub,
            #     center_region=center_region,
            # )

            if filter_circle is None:
                remove_inds.append(i_cluster)
                continue  # no fit found
            hough_ransac_circles.append(filter_circle)

            # compute fitness score for circle fit
            dists_center = np.linalg.norm(
                points[:, :2] - filter_circle.center[:2], axis=1
            )
            dists_circle = np.abs(dists_center - filter_circle.radius)
            score = np.sum(dists_circle < 0.05 * filter_circle.radius) / np.sum(
                dists_center < 1.05 * filter_circle.radius
            )
            scores.append(score)
            point_count = np.sum(dists_circle < 0.05 * filter_circle.radius)
            point_counts.append(point_count)
            filter_indices.append(i_cluster)

        # remove filtered clusters
        for i in remove_inds[::-1]:
            sliced_clusters.pop(i)
        # no fit found
        if len(hough_ransac_circles) == 0:
            return None
        # bad fit
        if np.mean(scores) < 0.33333 or np.sum(scores) < 1e-12:
            return None

        # filter points using the hough_ransac circle
        remove_inds = []
        for i_cluster in range(len(sliced_clusters)):
            filter_circle: Circle = hough_ransac_circles[i_cluster]
            filter_mask = (
                filter_circle.get_distance(sliced_clusters[i_cluster]) < filter_radius
            )
            pc_filtered = sliced_clusters[i_cluster][filter_mask]
            if len(pc_filtered) < 10:
                remove_inds.append(i_cluster)
            sliced_clusters[i_cluster] = pc_filtered
        for i in remove_inds[::-1]:
            sliced_clusters.pop(i)
            hough_ransac_circles.pop(i)
            scores.pop(i)
            point_counts.pop(i)
            filter_indices.pop(i)
        if len(sliced_clusters) == 0:
            return None  # no slices left
        if np.sum(point_counts) < 10:
            return None  # to few points left for fitting

        return {
            "hough_ransac_circles": hough_ransac_circles,
            "scores": np.array(scores),
            "point_counts": np.array(point_counts),
            "filtered_points": deepcopy(sliced_clusters),
            "filter_indices": np.array(filter_indices),
        }

    def reconstruct(  # noqa: C901, this function is complicated ^^
        self,
        max_height: float = None,
        slice_heights: Union[float, Iterable] = 1.0,
        slice_thickness: float = 0.2,
        max_center_deviation: float = 0.1,
        max_radius: float = 1.0,
        filter_radius: float = 0.05,
        max_consecutive_fails: int = 5,
        min_tree_height: float = 5,
    ) -> bool:
        """reconstructs the set of clusters into a stack of cone frustums represented by
        a stack of circles.

        Args:
            thisfunctioniscomplicated (_type_): _description_
            max_height (float, optional): Maximum height until which reconstructions are
                attempted. Defaults to None.
            slice_heights (Union[float, Iterable], optional): Either an increment or a
                set of slice heights. Defaults to 1.0.
            slice_thickness (float, optional): Thickness of a slice. Defaults to 0.2.
            max_center_deviation (float, optional): maximum deviation of a circle with
                respect to the circle beneath it. Defaults to 0.1.
            max_radius (float, optional): maximum radius of tree. Reducing this
                greatly reduces compute time. Defaults to 1.0.
            filter_radius (float, optional): Distance used for filtering to consider a
                point represented by a circle model. Defaults to 0.05.
            max_consecutive_fails (int, optional): stopping criterion for maximum
                consecutive failed attemptes. Defaults to 5.
            min_tree_height (float, optional): reconstructions smaller than this are
                discarded. Defaults to 5.

        Returns:
            bool: True, if reconstruction succeeds, False otherwise
        """

        circle_stack = []
        self.load_points()
        cluster_points_map = [
            cluster.cloud.point.positions.numpy()
            @ cluster.info.T_sensor2map[:3, :3].T
            + cluster.info.T_sensor2map[:3, 3]
            for cluster in self.clusters
        ]
        self.store_points()
        cluster_points_upright = [
            (points - self.axis.transform[:3, 3]) @ self.axis.transform[:3, :3]
            for points in cluster_points_map
        ]
        # filter out far points
        cluster_points_upright = [
            cpu[np.linalg.norm(cpu[:, :2], axis=1) < 2 * max_radius]
            for cpu in cluster_points_upright
        ]
        
        if np.any([len(cpu) == 0 for cpu in cluster_points_upright]):
            return False
        
        if type(slice_heights) == float:
            min_point = np.min([cpu.min() for cpu in cluster_points_upright])
            max_point = np.max([cpu.max() for cpu in cluster_points_upright])
            slice_heights = np.arange(
                min_point,
                max_height if max_height is not None else max_point,
                slice_heights,
            )
            if len(slice_heights) < 3:
                return False  # too few slices

        fail_counter = 0  # counts how many slices failed to fit a circle
        init_candidates = []
        num_init_candidates = 3
        i_slice_height = 0

        while i_slice_height < len(slice_heights):  # len of slice_heights is changed
            slice_height = slice_heights[i_slice_height]
            if fail_counter == max_consecutive_fails:
                break

            # determine boundary conditions for hough_ransac fits
            if len(circle_stack) == 0:
                center_region = None
                radius_lb = 0.8 * self.axis.radius
                radius_ub = min(
                    3 * self.axis.radius, max_radius if max_radius else np.inf
                )
            else:
                center_region = Circle(circle_stack[-1].center, max_center_deviation)
                radius_lb = 0.9 * circle_stack[-1].radius
                radius_ub = min(
                    1.05 * circle_stack[-1].radius, max_radius if max_radius else np.inf
                )

            # slice clusters and remove points further than twice the radius away from
            # the axis
            sliced_clusters = [
                cluster_points[
                    np.logical_and(
                        cluster_points[:, 2] >= slice_height - slice_thickness / 2,
                        cluster_points[:, 2] < slice_height + slice_thickness / 2,
                    )
                ]
                for cluster_points in cluster_points_upright
            ]

            # try to fit hough ransac circles to the slices
            slice_circles = self._filter_slice(
                sliced_clusters,
                radius_lb,
                radius_ub,
                center_region,
                filter_radius,
                slice_height,
            )
            if slice_circles is None:
                fail_counter += 1
                i_slice_height += 1
                continue

            # fit bulloc circles with fixed radius of mean hough_ransac circles
            weights = slice_circles["scores"] * slice_circles["point_counts"]
            if np.sum(weights) < 1e-12:
                fail_counter += 1
                i_slice_height += 1
                continue
            average_radius = np.average(
                [c.radius for c in slice_circles["hough_ransac_circles"]],
                weights=weights,
            )
            slice_circles["bullock_circles"] = [
                Circle.from_cloud_bullock(
                    pc, fixed_radius=average_radius, circle_height=slice_height
                )
                for pc in slice_circles["filtered_points"]
            ]

            # shift all circles to using the bulloc circles
            mean_center = np.average(
                np.vstack([c.center for c in slice_circles["bullock_circles"]]),
                axis=0,
            )
            shifts = mean_center - np.vstack(
                [c.center for c in slice_circles["bullock_circles"]]
            )
            for i in range(len(slice_circles["filtered_points"])):
                slice_circles["filtered_points"][i] += shifts[i]

            # fit final bullock circle
            points = np.vstack(slice_circles["filtered_points"])
            weights = np.ones(len(points))
            slice_circles["final_circle"] = Circle.from_cloud_bullock(
                points, circle_height=slice_height
            )

            # reject bad fits
            if np.any(np.isnan(slice_circles["final_circle"].center)):
                fail_counter += 1
                i_slice_height += 1
                continue
            if slice_circles["final_circle"].radius > max_radius:
                fail_counter += 1
                i_slice_height += 1
                continue
            if (
                len(circle_stack)
                and slice_circles["final_circle"].radius > 2 * circle_stack[-1].radius
            ):
                fail_counter += 1
                i_slice_height += 1
                continue

            # NMS for the first circle
            if len(circle_stack) == 0:
                if len(init_candidates) == 0:
                    if i_slice_height == len(slice_heights) - 1:
                        return False  # no init can be found, too few good slices
                    # add additional slice heights for NMS candidates here
                    candidate_heights = np.linspace(
                        slice_heights[i_slice_height],
                        slice_heights[i_slice_height + 1],
                        num_init_candidates + 1,
                    )[1:-1]
                    for candidate_height in candidate_heights[::-1]:
                        slice_heights = np.insert(
                            slice_heights, i_slice_height + 1, candidate_height
                        )
                if len(init_candidates) < num_init_candidates:
                    # aggregate results
                    init_candidates.append(
                        {
                            "circle": slice_circles["final_circle"],
                            "scores": slice_circles["scores"],
                            "point_counts": slice_circles["point_counts"],
                        }
                    )
                    i_slice_height += 1
                if len(init_candidates) == num_init_candidates:
                    init_circles = []
                    init_scores = []
                    # evaluate results aka supress non-maxima
                    for init_candidate in init_candidates:
                        init_circles.append(slice_circles["final_circle"])
                        init_scores.append(
                            np.mean(
                                init_candidate["scores"]
                                * init_candidate["point_counts"]
                            )
                        )
                    # add the best candidate to the stack
                    circle_stack.append(init_circles[np.argmax(init_scores)])
                    # print(f"found init, using {np.argmax(init_scores)}")
                    i_slice_height += 1
            else:
                circle_stack.append(slice_circles["final_circle"])
                i_slice_height += 1
            fail_counter = 0

        if len(circle_stack) < 2:
            self.reconstructed = False
            return False

        if circle_stack[-1].center[2] - circle_stack[0].center[2] < min_tree_height:
            return False

        self.circles = circle_stack
        self.reconstructed = True
        self.transform_circles(
            self.axis.transform[:3, 3], self.axis.transform[:3, :3]
        )
        return True

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """generates a mesh for the tree by connecting the circles with cone frustums

        Returns:
            np.ndarray: vertices of the mesh
            np.ndarray: triangle indices of the mesh
        """
        if self.reconstructed:
            vertices, triangles = np.empty((0, 3)), np.empty((0, 3), dtype=int)
            for i in range(len(self.circles) - 1):
                verts, tris = self.circles[i].genereate_cone_frustum_mesh(
                    self.circles[i + 1]
                )
                triangles = np.vstack((triangles, tris + vertices.shape[0]))
                vertices = np.vstack((vertices, verts))

            return vertices, triangles
        else:
            lower_center = self.axis.transform[:3, 3]
            cylinder_radius = self.axis.radius
            cylinder_axis = self.axis.transform[:3, 2]
            cylinder_height = self.place_holder_height
            upper_center = lower_center + cylinder_axis * cylinder_height

            bottom_circle = Circle(lower_center, cylinder_radius, cylinder_axis)
            top_circle = Circle(upper_center, cylinder_radius, cylinder_axis)

            retval = bottom_circle.genereate_cone_frustum_mesh(top_circle)
            mesh = trimesh.Trimesh(vertices=retval[0], faces=retval[1])
            return mesh.vertices, mesh.faces

    def generate_canopy(self, min_floor_height: float = 2.0) -> bool:
        """Computes a canopy mesh from the points. Canopy points are points, not closer
        to the ground than min_floor_height and furhter from the stem axis than 1.5 the
        local radius.

        Args:
            min_floor_height (float, optional): Points further from the floor than this
                and far enough from the stem are considered canopy points.
                Defaults to 2.0.

        Returns:
            bool: True if canopy volume could be found.
        """
        if not self.reconstructed:
            return False
        # filter out points close to floor (by map z axis)
        canopy_points = self.points[
            self.points[:, 2] - np.min(self.points[:, 2]) > min_floor_height
        ]
        if len(canopy_points) < 10:
            return False
        # filter out points close to reconstruction
        axis = np.concatenate(
            (self.axis.transform[:3, 2], self.axis.transform[:3, 3])
        )
        sq_dists = pnts_to_axes_sq_dist(canopy_points, axis[None, :]).flatten()
        mask = np.logical_or(
            sq_dists > (1.5 * self.axis.radius) ** 2,
            canopy_points[:, 2] > np.max(canopy_points[:, 2]) - 0.5,
        )
        canopy_points = canopy_points[mask]

        if len(canopy_points) < 10:
            return False

        # make o3d point cloud from canopy points
        pcd = o3d.t.geometry.PointCloud(canopy_points)
        hull = pcd.compute_convex_hull()
        self.canopy_mesh = {
            "vertices": hull.vertex.positions.numpy(),
            "triangles": hull.triangle.indices.numpy(),
        }

    def compute_dbh(self, terrain_height: float) -> None:
        """Computes the diameter at breast height (1.3 m above the terrain height)

        Args:
            terrain_height (float): terrain height at location of tree. This must be
                computed using a terrain model, which is not part of this tree object.
        """
        reco_dbh = None
        query_height = terrain_height + 1.3
        dbh_circle = self.evaluate_at_height(query_height)
        if dbh_circle is not None:
            reco_dbh = dbh_circle.radius * 2
            self.dbh = reco_dbh

    def __str__(self) -> str:
        """Converts this object into a string representation

        Returns:
            str: String descriping this tree
        """
        if self.circles is not None:
            return f"Tree {self.id} with {len(self.circles)} circles and {len(self.clusters)} clusters"
        else:
            return f"Tree {self.id} with no circles and {len(self.clusters)} clusters"

    def remove_tmp_file(self):
        """Removes all temporary files from disk used for storing points."""
        if self._tmp_path is None:
            return
        filename = os.path.join(
            self._tmp_path, "tree_points", f"tree_{self.id:0>5}.pkl"
        )
        if os.path.exists(filename):
            os.remove(filename)

    def get_export_dict(self) -> dict:
        """This dict contains all relevant information of the tree. It is used for
        exporting the tree to disk.

        Returns:
            dict: Dictionary containing all relevant information of the tree
        """
        export_dict = {
            "id": self.id,
            "place_holder_height": self.place_holder_height,
            "reconstructed": self.reconstructed,
            "circles": self.circles,
            "canopy_mesh": self.canopy_mesh,
            "clusters": deepcopy(self.clusters),
            "dbh": self.dbh,
        }
        for cluster in export_dict["clusters"]:
            del cluster.cloud
        return export_dict

    def write_to_disk(self, root_dir: str) -> None:
        """Writes all information to disc including point cloud as pcd files and
        tree traits and addtional variables structure as pickle file.

        Args:
            root_dir (str): dirctory to write to
        """
        tree_dir = os.path.join(root_dir, f"tree_{self.id:0>5}")
        if not os.path.exists(tree_dir):
            os.makedirs(tree_dir)
        else:
            for file in os.listdir(tree_dir):
                os.remove(os.path.join(root_dir, f"tree_{self.id:0>5}", file))
        with open(os.path.join(tree_dir, "tree.pkl"), "wb") as file:
            pickle.dump(self.get_export_dict(), file)
        for cluster in self.clusters:
            try:
                cloud = cluster.cloud.to_legacy()
            except AttributeError:
                # cloud already is legacy
                cloud = cluster.cloud
            seconds, nanoseconds = cluster.info.time_stamp.seconds_nanoseconds()
            o3d.io.write_point_cloud(
                os.path.join(
                    tree_dir,
                    f"cluster_{seconds}_{nanoseconds:0>9}.pcd",
                ),
                cloud,
            )

    def __del__(self):
        """Destructor of the Tree object. It removes all temporary files from disk."""
        self.remove_tmp_file()
