from colorsys import hls_to_rgb, hsv_to_rgb
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Tuple
from realtime_trees.utils.ros_proxys import Time

import numpy as np
import open3d as o3d
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from realtime_trees.circle import Circle
from realtime_trees.utils.timing import Timer
from realtime_trees.utils.meshing import meshgrid_to_mesh
from realtime_trees.utils.distances import pnts_to_axes_sq_dist
from realtime_trees.utils.dataclasses import ClusterAxis, CoverageInfo, ClusterInfo, Cluster


timer = Timer()
# globals for debugging
current_point_cloud = 0
current_view = None

def _compute_labels_axes_voronoi(  # noqa: C901
    cloud: o3d.t.geometry.PointCloud,
    cloth: np.ndarray = None,
    crop_bounds: list = [[0.5, 1.5], [2, 3], [4, 5]],
    max_cluster_radius: float = np.inf,
    max_tree_radius: float = 1.0,
    n_threads: int = 1,
    precise_calcuation_fraction: float = 0.1,
    dbscan_eps: float = 0.7,
    debug_level: int = 0,
) -> Tuple[np.ndarray, List[ClusterAxis]]:
    """This function clusters the point cloud into tree instances using a combination of
    DBSCAN clustering, the Hough-ransac algorithm and Voronoi tesselation. The function
    returns the labels of the clusters and the axes of the trees. The label -1 is used for
    points that are not part of a tree.

    Args:
        cloth (np.ndarray, optional): Terrain model for normalizing the point cloud.
            It should be given as the meshgrid format provided by
            realtime_trees.terrain.fit_terrain. Defaults to None.
        crop_bounds (list, optional): list of cropping intervals. In these intervals,
            cylinders are searched and NMS is used to select the best fit.
            Defaults to [[0.5, 1.5], [2, 3], [4, 5]].
        max_cluster_radius (float, optional): maximum radius of a cluster. This should
            be larger than the canopy radius. Defaults to np.inf.
        max_tree_radius (float, optional): maximum expected tree radius. 
            Defaults to 1.0 m.
        n_threads (int, optional): When not equal to 1, multiprocessing is used to
            calculate the distances from points to axes. This can be useful for very
            large point clouds. However, it is more efficient and almost no less
            accurate to calculate the distances for only few points
            (precise_calcuation_fraction of all poitns) and to find the rest of the
            labels by inheriting the label of the closest neighbor . Defaults to 1.
        precise_calcuation_fraction (float, optional): Only this fraction of the points
            is used to calculate the precise distance to the axes. The rest of the
            points inherit label and distance from the closest point with a label.
            Defaults to 0.1.
        dbscan_eps (float, optional): The epsilon parameter for DBSCAN clustering.
            Defaults to 0.7.
        debug_level (int, optional): Level of debugging for print messages.
            Defaults to 0.

    Returns:
        np.ndarray, List[ClusterAxis]: labels for every point and a list of ClusterAxis
            object describing the main axes of the trees.
    """
    final_labels = -np.ones(cloud.point.positions.shape[0], dtype=np.int32)

    # 1. Normalize heights
    with timer("normalizing heights"):
        if cloth is not None:
            height_interpolator = RegularGridInterpolator(
                points=(cloth[:, 0, 0], cloth[0, :, 1]),
                values=cloth[:, :, 2],
                method="linear",
                bounds_error=False,
                fill_value=0.0,
            )
            heights = height_interpolator(cloud.point.positions.numpy()[:, :2])
            cloud_orig = cloud.clone()
            cloud.point.positions[:, 2] -= heights.astype(np.float32)
    if debug_level > 1:
        print("VIZ: Height-normalized Cloud")

        def toggle_point_cloud(vis):
            global current_point_cloud
            current_view = vis.get_view_control().convert_to_pinhole_camera_parameters()
            verts, tris = meshgrid_to_mesh(cloth)
            verts_vec = o3d.utility.Vector3dVector(verts)
            tris_vec = o3d.utility.Vector3iVector(
                np.concatenate((tris, np.flip(tris, axis=1)), axis=0)
            )
            terrain_mesh = o3d.geometry.TriangleMesh(verts_vec, tris_vec)
            if current_point_cloud == 1:
                vis.clear_geometries()
                vis.add_geometry(terrain_mesh)
                vis.add_geometry(cloud_orig.to_legacy())
                current_point_cloud = 2
                print("Showing Unnormalized cloud")
            else:
                vis.clear_geometries()
                vis.add_geometry(cloud.to_legacy())
                print("Showing Normalized cloud")
                current_point_cloud = 1
            vis.get_view_control().convert_from_pinhole_camera_parameters(
                current_view, True
            )

        global current_point_cloud
        current_point_cloud = 1
        visualizer = o3d.visualization.VisualizerWithKeyCallback()
        visualizer.create_window()
        visualizer.add_geometry(cloud.to_legacy())
        visualizer.register_key_callback(ord("T"), toggle_point_cloud)
        visualizer.run()
        visualizer.destroy_window()

    # 2. Crop point cloud between cluster_strip_min and cluster_strip_max
    with timer("cropping"):
        points_numpy = cloud.point.positions.numpy()
        crops = []
        for bounds in crop_bounds:
            mask = np.logical_and(
                points_numpy[:, 2] > bounds[0], points_numpy[:, 2] < bounds[1]
            )
            crop = cloud.select_by_mask(mask.astype(bool))
            crops.append(crop)

    if debug_level > 1:
        print("VIZ: Cropped cloud for clustering")
        o3d.visualization.draw_geometries([c.to_legacy() for c in crops])

    # 3. Perform db scan clustering after removing outliers
    with timer("dbscan clustering of crops"):
        crop_labels = []
        for i in range(len(crops)):
            crop = crops[i]
            _, ind = crop.to_legacy().remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0
            )
            if len(ind):
                crops[i] = crop.select_by_index(ind)
                crop_labels.append(
                    crops[i]
                    .cluster_dbscan(eps=dbscan_eps, min_points=20, print_progress=False)
                    .numpy()
                )
            else:
                crops[i] = o3d.t.geometry.PointCloud()
                crop_labels.append(np.array([]))

    if debug_level > 1:
        print("VIZ: DBSCAN Clustering")
        clusters = []
        for c, l in zip(crops, crop_labels):
            max_label = np.max(l)
            for i in range(max_label):
                cluster_points = c.select_by_mask(l == i)
                cluster_points.paint_uniform_color(np.random.rand(3))
                clusters.append(cluster_points.to_legacy())
        o3d.visualization.draw_geometries(clusters)

    # 4. Clean up non-stem points using hough transform
    with timer("hough"):
        axes: List[ClusterAxis] = []
        for c, label, bounds in zip(crops, crop_labels, crop_bounds):
            if len(label) == 0:
                continue
            max_label = np.max(label)
            for i_label in range(max_label):
                cluster_points = c.select_by_mask(label == i_label)
                cluster_points = cluster_points.point.positions.numpy()
                if cluster_points.shape[0] < 50:
                    if debug_level > 0:
                        print(
                            f"Cluster {i_label} has only {cluster_points.shape[0]} points. Skipping"
                        )
                    continue

                # Remove clusters not extending between bounds
                if (
                    np.min(cluster_points[:, 2]) > 1.05 * bounds[0]
                    or np.max(cluster_points[:, 2]) < 0.95 * bounds[1]
                ):
                    if debug_level > 0:
                        print(
                            f"Cluster {i_label} does not extend between bounds. Skipping"
                        )
                    continue
                # Find circles in slice of crop at middle
                slice_height = 0.2  # m
                slice_lower = cluster_points[
                    np.logical_and(
                        cluster_points[:, 2] > bounds[0],
                        cluster_points[:, 2] < bounds[0] + slice_height,
                    )
                ]
                slice_upper = cluster_points[
                    np.logical_and(
                        cluster_points[:, 2] < bounds[1],
                        cluster_points[:, 2] > bounds[1] - slice_height,
                    )
                ]
                hough_kwargs = dict(
                    grid_res=0.02,
                    min_radius=0.05,
                    max_radius=max_tree_radius,
                    return_pixels_and_votes=True,
                )

                circ_lower = Circle.from_cloud_hough_ransac(
                    points=slice_lower,
                    **hough_kwargs,
                    circle_height=bounds[0],
                    max_points=50,
                )
                circ_upper = Circle.from_cloud_hough_ransac(
                    points=slice_upper,
                    **hough_kwargs,
                    circle_height=bounds[1],
                    max_points=50,
                )
                if circ_lower is None or circ_upper is None:
                    if debug_level > 0:
                        print(f"No votes for cluster {i_label}. Skipping")
                    continue

                if debug_level > 0:
                    print("found two hough circles")

                cylinder_radius = (circ_lower.radius + circ_upper.radius) / 2
                T = np.eye(4)
                tree_axis = circ_upper.center - circ_lower.center
                tree_axis /= np.linalg.norm(tree_axis)

                # large trees are expected to be upright!
                if cylinder_radius > 0.8 * hough_kwargs["max_radius"]:
                    max_angle = 10
                else:
                    max_angle = 20
                if np.rad2deg(np.arccos(tree_axis[2])) > max_angle:
                    if debug_level > 0:
                        print(f"Cluster {i_label} is not vertical enough. Skipping")
                    continue

                # Compute Score of circle fit: points insde / points up to r away from circle
                dists = (
                    pnts_to_axes_sq_dist(
                        cluster_points,
                        np.concatenate((tree_axis, circ_lower.center))[None, :],
                        apply_sqrt=True,
                    )
                    - cylinder_radius
                )
                score = np.sum(np.abs(dists) < 0.1 * cylinder_radius) / np.sum(
                    dists < 0.1 * cylinder_radius
                )

                if score < 0.25:
                    if debug_level > 0:
                        print(f"Cluster {i_label} has a bad score ({score}). Skipping")
                    continue

                axis_normal = np.array([tree_axis[1], -tree_axis[0], 0])
                axis_normal /= np.linalg.norm(axis_normal)
                T[:3, :3] = np.stack(
                    (axis_normal, np.cross(tree_axis, axis_normal), tree_axis), axis=1
                )
                T[:3, 3] = np.array(
                    [circ_lower.center[0], circ_lower.center[1], bounds[0]]
                )
                cluster_axis = ClusterAxis(
                    transform=T,
                    radius=cylinder_radius,
                    height=bounds[1] - bounds[0],
                    score=score
                )
                axes.append(cluster_axis)

    if len(axes) == 0:
        return np.array([]), []
        
    # Do Non-maximum suppression
    with timer("non-maximum suppression"):
        nms_radius = 1.0
        kdtree = cKDTree(np.array([a.transform[:2, 3] for a in axes]))
        # find all clusters of radius nms_radius
        indices = kdtree.query_ball_tree(kdtree, nms_radius)
        if max(len(i) for i in indices) > len(crops):
            print("WARNING: NMS radius too large. Some clusters are over-suppressed")

        unique_sets = set([tuple(sorted(i)) for i in indices])
        nms_result = [max(us, key=lambda i: axes[i].score) for us in unique_sets]
        axes_nms = [axes[i] for i in nms_result]

    if debug_level > 1:
        print("VIZ: DBSCAN Clustering and Fitted Axes")
        dbscan_clusters = []
        for c, label in zip(crops, crop_labels):
            max_label = np.max(label)
            for i in range(max_label):
                cluster_points = c.select_by_mask(label == i)
                cluster_points.paint_uniform_color(np.random.rand(3))
                dbscan_clusters.append(cluster_points.to_legacy())

        cylinders = []
        cylinders_nms = []
        for axis in axes:
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                radius=axis.radius, height=axis.height
            )
            # shift up or down depending on third component of pca. Thus the
            # cylinder allways covers the pc
            cylinder.vertices = o3d.utility.Vector3dVector(
                np.array(cylinder.vertices) + np.array([0, 0, axis.height / 2])
            )
            if axis.score > 0.25:
                cylinder.paint_uniform_color([axis.score, axis.score / 2, 0])
            else:
                cylinder.paint_uniform_color([1, 0, 0])
            cylinder.vertices = o3d.utility.Vector3dVector(
                (np.array(cylinder.vertices) @ axis.transform[:3, :3].T)
                + axis.transform[:3, 3]
            )
            cylinders.append(cylinder)
        for axis in axes_nms:
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                radius=axis.radius, height=axis.height * 1.2
            )
            # shift up or down depending on third component of pca. Thus the
            # cylinder allways covers the pc
            cylinder.vertices = o3d.utility.Vector3dVector(
                np.array(cylinder.vertices) + np.array([0, 0, axis.height / 2])
            )
            cylinder.paint_uniform_color([0, 1, 0])
            cylinder.vertices = o3d.utility.Vector3dVector(
                (np.array(cylinder.vertices) @ axis.transform[:3, :3].T)
                + axis.transform[:3, 3]
            )
            cylinders_nms.append(cylinder)

        o3d.visualization.draw_geometries(
            [cloud.to_legacy()] + cylinders + cylinders_nms
        )
    axes = axes_nms

    # calculate distance to each axis
    with timer("voronoi"):
        axes_np = np.array(
            [np.hstack((a.transform[:3, 2], a.transform[:3, 3])) for a in axes]
        )
        if precise_calcuation_fraction < 1.0:
            precise_mask = (
                np.random.rand(points_numpy.shape[0]) < precise_calcuation_fraction
            )
            precise_query_points = points_numpy[precise_mask]
        else:
            precise_query_points = points_numpy

        if n_threads == 1:
            precise_dists = pnts_to_axes_sq_dist(
                points=precise_query_points,
                axes=axes_np,
                debug_level=debug_level,
            )
        else:
            with Pool() as pool:
                points_grouped = np.array_split(precise_query_points, n_threads, axis=0)
                dists = pool.map(
                    partial(
                        pnts_to_axes_sq_dist, axes=axes_np, debug_level=debug_level
                    ),
                    points_grouped,
                )
                precise_dists = np.vstack(dists)

        precise_labels = np.argmin(precise_dists, axis=1)
        precise_min_dists = precise_dists[
            np.arange(precise_dists.shape[0]), precise_labels
        ]
        if precise_calcuation_fraction < 1.0:
            final_labels = np.empty((points_numpy.shape[0]), dtype=np.int32)
            min_dists = np.empty((points_numpy.shape[0]))
            # fill precise values
            final_labels[precise_mask] = precise_labels
            min_dists[precise_mask] = precise_min_dists
            # find other values using cKD tree
            kd_tree = cKDTree(points_numpy[precise_mask])
            _, idcs = kd_tree.query(points_numpy[~precise_mask], k=1, workers=-1)
            final_labels[~precise_mask] = precise_labels[idcs]
            min_dists[~precise_mask] = precise_min_dists[idcs]
        else:
            dists = precise_dists
            final_labels = precise_labels
            min_dists = precise_min_dists

        if debug_level > 0:
            print("Clustering done")

        if max_cluster_radius != np.inf:
            final_labels[min_dists > max_cluster_radius**2] = -1

    with timer("data grooming"):
        # remove clusters with fewer than 50 points
        filtered_axes = []
        unique_labels, counts = np.unique(final_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            if count < 50:
                final_labels[final_labels == label] = -1
        # make sure the label index is continuous.
        unique_labels = np.sort(np.unique(final_labels))
        for i, label in enumerate(unique_labels[1:]):
            final_labels[final_labels == label] = i
            filtered_axes.append(axes[label])
        # denormalize heights in clusters
        if cloth is not None:
            tree_centers = np.array([a.transform[:2, 3] for a in filtered_axes])
            terrain_heights = height_interpolator(tree_centers)
            for i, axis in enumerate(filtered_axes):
                axis.transform[2, 3] += terrain_heights[i]

    if debug_level > 1:
        print("VIZ: Voronoi Clustering")
        clusters = []
        for label in np.unique(final_labels):
            if label == -1:
                continue
            cluster_points = cloud.select_by_mask(final_labels == label)
            cluster_points.paint_uniform_color(hls_to_rgb(np.random.rand(), 0.6, 1.0))
            clusters.append(cluster_points.to_legacy())
        o3d.visualization.draw_geometries(clusters + cylinders_nms + cylinders)
    if debug_level > 0:
        print(timer)

    # renormalize heights
    if cloth is not None:
        cloud.point.positions[:, 2] += heights.astype(np.float32)

    return final_labels, filtered_axes


def _prefiltering(
    cloud: o3d.t.geometry.PointCloud, normal_thr: float = 0.5, voxel_size: float = 0.05
) -> o3d.t.geometry.PointCloud:
    """prefilters the point cloud by removing the floor using the point normals and
    voxel-downsampling

    Args:
        cloud (o3d.t.geometry.PointCloud): input point cloud
        normal_thr (float, optional): threshold of the z component of the point normals.
            Defaults to 0.5.
        voxel_size (float, optional): size for the voxel filter. Defaults to 0.05.

    Returns:
        o3d.t.geometry.PointCloud: _description_
    """
    # Filter by Z-normals
    mask = (cloud.point.normals[:, 2] >= -normal_thr) & (
        cloud.point.normals[:, 2] <= normal_thr
    )
    new_cloud = cloud.select_by_mask(mask)

    # Downsample
    # new_cloud = new_cloud.voxel_down_sample(voxel_size=voxel_size) # faster way
    new_cloud, _, _ = new_cloud.to_legacy().voxel_down_sample_and_trace(
        voxel_size,
        new_cloud.get_min_bound().numpy().astype(np.float64),
        new_cloud.get_max_bound().numpy().astype(np.float64),
    )
    new_cloud = o3d.t.geometry.PointCloud.from_legacy(new_cloud)

    return new_cloud


def cluster(
    cloud: o3d.t.geometry.PointCloud,
    terrain: np.ndarray = None,
    crop_bounds: list = [[0.5, 1.5], [2, 3], [4, 5]],
    max_cluster_radius: float = 5.0,
    max_tree_radius: float = 1.0,
    n_threads: int = 1,
    precise_calcuation_fraction: float = 0.1,
    debug_level: int = 0,
    normal_thr: float = 0.5,
    voxel_size: float = 0.05,
    dbscan_eps: float = 0.7,
) -> List[Cluster]:
    """Clusters the point cloud into tree instances

    Args:
        cloud (o3d.t.geometry.PointCloud): Input point cloud
        cloth (np.ndarray, optional): Terrain model for normalizing the point cloud.
            It should be given as the meshgrid format provided by
            realtime_trees.terrain.fit_terrain. Defaults to None.
        crop_bounds (list, optional): list of cropping intervals. In these intervals,
            cylinders are searched and NMS is used to select the best fit.
            Defaults to [[0.5, 1.5], [2, 3], [4, 5]].
        max_cluster_radius (float, optional): maximum radius of a cluster. This should
            be larger than the canopy radius. Defaults to 5.0 m.
        max_tree_radius (float, optional): maximum expected tree radius. This is used
            to set the NMS radius. Defaults to 1.0 m.
        n_threads (int, optional): When not equal to 1, multiprocessing is used to
            calculate the distances from points to axes. This can be useful for very
            large point clouds. However, it is more efficient and almost no less
            accurate to calculate the distances for only few points
            (precise_calcuation_fraction of all poitns) and to find the rest of the
            labels by inheriting the label of the closest neighbor . Defaults to 1.
        precise_calcuation_fraction (float, optional): Only this fraction of the points
            is used to calculate the precise distance to the axes. The rest of the
            points inherit label and distance from the closest point with a label.
            Defaults to 0.1.
        debug_level (int, optional): Level of debugging for print messages.
            Defaults to 0.
        normal_thr (float, optional): threshold of the z component of the point normals.
            Defaults to 0.5.
        voxel_size (float, optional): size for the voxel filter. Defaults to 0.05.
        dbscan_eps (float, optional): The epsilon parameter for DBSCAN clustering.
            Defaults to 0.7.

    Returns:
        List[Cluster]: List of clusters with point cloud and info
    """
    # prefilter
    cloud = _prefiltering(cloud, normal_thr, voxel_size)

    # find tree axes and associate points to closest axis
    labels, axes = _compute_labels_axes_voronoi(
        cloud,
        terrain,
        crop_bounds,
        max_cluster_radius,
        max_tree_radius,
        n_threads,
        precise_calcuation_fraction,
        dbscan_eps,
        debug_level,
    )
    
    if len(labels) == 0:
        return []

    # parse data
    clusters = []
    num_labels = labels.max() + 1
    for i in range(num_labels):
        mask = labels == i
        seg_cloud = cloud.select_by_mask(mask)
        color = hsv_to_rgb(np.random.rand(), 0.6, 1.0)
        cluster = Cluster(
            cloud=seg_cloud,
            info=ClusterInfo(id=i, color=color, axis=axes[i]),
        )
        clusters.append(cluster)

    if debug_level > 0:
        print("Extracted " + str(len(clusters)) + " initial clusters.")

    return clusters
