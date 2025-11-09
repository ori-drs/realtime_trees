import multiprocessing
from typing import List

import numpy as np

from realtime_trees.utils.dataclasses import ClusterAxis


def pnts_to_axes_sq_dist(
    points: np.ndarray,
    axes: np.ndarray,
    apply_sqrt: bool = False,
    debug_level: int = 0,
) -> np.ndarray:
    """Calculate the distance of all point to all axes. For efficiency, two planes are
    constructed fore every axis, which is the intersection of them.
    The distance of a point to the axis is then the L2 norm of the individual distances
    to both planes. This is ~5 times faster than using the cross product.

    Args:
        pnt (np.ndarray[Nx3]): point in 3D space
        axis (np.ndarray[Mx6]): axis in 3D space (direction vector, point on axis)
        sqrt (bool, optional): whether to return the sqrt of the squared distance.
            Defaults to False.
        point_fraction (float, optional): fraction of points to use for
            calculating the distance. The rest is determined by point to point distance
            calculation. Defaults to 0.1.
        debug_level (int, optional): verbosity level. Defaults to 0.

    Returns:
        np.ndarray[NxM]: (squared) distance of all points to all axes
    """
    if debug_level > 0:
        print("Start calculating distances on ", multiprocessing.current_process().name)
    axes = axes.copy()
    axis_dirs = axes[:, :3]
    axis_dirs /= np.linalg.norm(axis_dirs, axis=1, keepdims=True)
    axis_pnts = axes[:, 3:]
    # TODO handle case where axis direction is in x-y-plane
    # (extremely unlikely for digiforest)
    normals_a = np.vstack(
        [np.zeros_like(axis_dirs[:, 0]), axis_dirs[:, 2], -axis_dirs[:, 1]]
    ).T
    normals_a /= np.linalg.norm(normals_a, axis=1)[:, None]
    normals_b = np.cross(axis_dirs, normals_a)

    # hesse normal form in einstein notation
    axis_pnts_to_origin_a = np.einsum("ij,ij->i", axis_pnts, normals_a)
    axis_pnts_to_origin_b = np.einsum("ij,ij->i", axis_pnts, normals_b)
    signed_dist_a = np.einsum("ij,kj->ik", points, normals_a) - axis_pnts_to_origin_a
    signed_dist_b = np.einsum("ij,kj->ik", points, normals_b) - axis_pnts_to_origin_b
    # this is much faster than np.power and np.sum ?! ^^
    sq_dists = signed_dist_a * signed_dist_a + signed_dist_b * signed_dist_b

    if debug_level > 0:
        print("Done calculating distances on", multiprocessing.current_process().name)
    return np.sqrt(sq_dists) if apply_sqrt else sq_dists


def distance_line_to_line(
    line_1: ClusterAxis, line_2: ClusterAxis, clip_heights: List[float] = None
) -> float:
    """Calculates the minum distance between two axes. The closest points can be clipped
    between two global z values specified in clip_heights.

    Args:
        line1 (ClusterAxis): Dict with the keys "transform" and optionally
            "axis_length" describing the first axis.
        line2 (ClusterAxis): Dict with the keys "transform" and optionally
            "axis_length" describing the second axis.
        clip_heights (List[float], optional): If present, the z component of the closest points are
            bounded to be between global these heights in the frame of the axes.
            Defaults to None.

    Returns:
        float: minimum distance between axes
    """
    axis_pnt_1 = line_1.transform[:3, 3]
    axis_pnt_2 = line_2.transform[:3, 3]
    axis_dir_1 = line_1.transform[:3, 2]
    axis_dir_2 = line_2.transform[:3, 2]
    normal = np.cross(axis_dir_1, axis_dir_2)
    normal_length = np.linalg.norm(normal)
    if np.isclose(normal_length, 0.0):
        meeting_point_1 = axis_pnt_1
        # Part of Gram Schmidt
        meeting_point_2 = axis_pnt_1 - axis_dir_1 * (
            (axis_pnt_2 - axis_pnt_1) @ axis_dir_1
        )
    else:
        normal /= np.linalg.norm(normal_length)
        v_normal = np.cross(axis_dir_1, normal)
        v_normal /= np.linalg.norm(v_normal)
        w_normal = np.cross(axis_dir_2, normal)
        w_normal /= np.linalg.norm(w_normal)
        s = w_normal @ (axis_pnt_2 - axis_pnt_1) / (w_normal @ axis_dir_1)
        t = v_normal @ (axis_pnt_1 - axis_pnt_2) / (v_normal @ axis_dir_2)

        meeting_point_1 = axis_pnt_1 + s * axis_dir_1
        meeting_point_2 = axis_pnt_2 + t * axis_dir_2

    if clip_heights is not None:
        if meeting_point_1[2] < clip_heights[0]:
            meeting_point_1 -= (
                axis_dir_1 * (meeting_point_1[2] - clip_heights[0]) / axis_dir_1[2]
            )
        if meeting_point_1[2] > clip_heights[1]:
            meeting_point_1 -= (
                axis_dir_1 * (meeting_point_1[2] - clip_heights[1]) / axis_dir_1[2]
            )
        if meeting_point_2[2] < clip_heights[0]:
            meeting_point_2 -= (
                axis_dir_2 * (meeting_point_2[2] - clip_heights[0]) / axis_dir_2[2]
            )
        if meeting_point_2[2] > clip_heights[1]:
            meeting_point_2 -= (
                axis_dir_2 * (meeting_point_2[2] - clip_heights[1]) / axis_dir_2[2]
            )

    return np.linalg.norm(meeting_point_1 - meeting_point_2)
