from itertools import product
from typing import Union, Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from skimage.transform import hough_circle


class Circle:
    def __init__(
        self,
        center: Union[tuple, list, np.ndarray],
        radius: float,
        normal: np.ndarray = np.array([0, 0, 1]),
    ) -> None:
        """Intializes a circle object with center, radius and normal.

        Args:
            center (Union[tuple, list, np.ndarray]): center coordinates of circle
            radius (float): radius of circle
            normal (np.ndarray, optional): orientation of circle.
                Defaults to np.array([0, 0, 1]).
        """
        self.radius = radius
        if type(center) in (list, tuple):
            center = np.array(center)
        self.center = center

        y_vec = np.array([0, -normal[2], normal[1]]).astype(float)
        y_vec /= np.linalg.norm(y_vec)
        x_vec = np.cross(y_vec, normal)
        self.rot_mat = np.vstack((x_vec, y_vec, normal)).T

    @classmethod
    def from_cloud_hough(
        cls,
        points: np.ndarray,
        grid_res: float = 0.02,
        min_radius: float = 0.05,
        max_radius: float = 0.5,
        point_ratio: float = 0.0,
        center_region: "Circle" = None,
        entropy_weighting: float = 0.0,
        circle_height: float = 0.0,
        return_pixels_and_votes: bool = False,
        **kwargs,
    ) -> Tuple["Circle", np.ndarray, np.ndarray, np.ndarray]:
        """This function fits a circle to the points in a slice using the hough
        transform. If both previous_center and search_radius are given, the search
        for the circle center is constrained to a circle around the previous center.
        If previous_radius is given, the search for the circle radius is constrained
        to be smaller than this radius.

        Args:
            points (np.ndarray): N x 2 array of points in the slice
            grid_res (float): length of pixel for hough map in m
            min_radius (float): minimum radius of circle in m
            max_radius (float): maximum radius of circle in m
            point_ratio (float, optional): ratio of points in a pixel wrt. number in
                most populated pixel to be counted valid. Defaults to 1.0.
            previous_center (np.ndarray, optional): x,y coordinates of the previous
                circle center. Defaults to None.
            search_radius (float, optional): radius around previous center to search
            entropy_weighing (float, optional): weight to weigh the hough votes by
                the entropy of the top 10 votes for that radius. This helps to cope
                with noise at small radii. Defaults to 10.0.
            circle_height (float, optional): height of the slice in m.
                Defaults to 0.0.
            return_pixels_and_votes (bool, optional): If True, an array containing the
                pixels aggregating the points and the votes corresponding to the optimal
                radius casted are returned additionally. Useful for debugging.
                Defaults to False.

        Returns:
            Circle: circle object and if wanted the pixels aggregating the
                points and the entropy_weighted hough votes and the penalty factor.
                The unweighted votes can be obtained by multiplying weighted votes with
                the factor.
        """
        if len(points) == 0:
            if return_pixels_and_votes:
                return None, None, None, None
            else:
                return None

        # construct 2D grid with pixel length of grid_res
        # bounding the point cloud of the current slice
        min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
        min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
        # make sure pixels are square
        grid_center = np.array([(max_x + min_x) / 2, (max_y + min_y) / 2])
        grid_width = 1.5 * max(max_x - min_x, max_y - min_y)
        min_x, max_x = grid_center[0] - grid_width / 2, grid_center[0] + grid_width / 2
        min_y, max_y = grid_center[1] - grid_width / 2, grid_center[1] + grid_width / 2
        n_cells = int(grid_width / grid_res)

        if n_cells < 5:
            if return_pixels_and_votes:
                return None, None, None, None
            else:
                return None

        # construct the grid
        grid_x, grid_y = np.meshgrid(
            np.linspace(min_x, max_x, n_cells + 1),
            np.linspace(min_y, max_y, n_cells + 1),
        )
        # reshape and remove last row and column
        grid_x = grid_x[None, :-1, :-1]
        grid_y = grid_y[None, :-1, :-1]

        # count how many points are in every cell (numpy version consumes too much mem)
        pixels = np.zeros((n_cells, n_cells))
        for point in points:
            i = int((point[1] - min_y) / grid_res)
            j = int((point[0] - min_x) / grid_res)
            pixels[i, j] += 1
        pixels[pixels < point_ratio * np.max(pixels)] = 0

        # crop image to only include the pixels containing points with a 50% margin
        filled_pixels = np.argwhere(pixels)
        min_x, max_x = np.min(filled_pixels[:, 0]), np.max(filled_pixels[:, 0])
        min_y, max_y = np.min(filled_pixels[:, 1]), np.max(filled_pixels[:, 1])
        min_x = max(0, min_x - int(0.5 + 0.5 * (max_x - min_x)))
        max_x = min(n_cells, max_x + int(0.5 + 0.5 * (max_x - min_x)))
        min_y = max(0, min_y - int(0.5 + 0.5 * (max_y - min_y)))
        max_y = min(n_cells, max_y + int(0.5 + 0.5 * (max_y - min_y)))
        pixels = pixels[min_x:max_x, min_y:max_y]
        grid_x = grid_x[:, min_x:max_x, min_y:max_y]
        grid_y = grid_y[:, min_x:max_x, min_y:max_y]

        # fit circles to the points in every cell using the hough transform
        min_radius_px = int(0.5 + min_radius / grid_res)
        max_radius_px = int(0.5 + max_radius / grid_res)
        # assume that at least a quarter of the points are seen. Then the max radius
        # is the number of pixels in one direction
        max_radius_px = min(max_radius_px, n_cells)
        try_radii = np.arange(min_radius_px, max_radius_px)
        if try_radii.shape[0] == 0:
            if return_pixels_and_votes:
                return None, None, None, None
            else:
                return None
        hough_res = hough_circle(pixels, try_radii)

        # weigh each radius by the entropy of the top 10 hough votes for that radius
        # for small radii there are many circles in a small area leading to many
        # high values. Which might be higher than for higher radii, which we are
        # looking for more. To avoid this, we weigh each radius by the entropy of
        # the top 10 hough votes for that radius, to reward radii that have a clear
        # vote.
        if entropy_weighting != 0.0:
            # To avoid artefacts where large radii have cropped voting-rings that
            # lead to artificially high entropy, exclude radii where both the
            # following applies:
            #    1. the radius is bigger than 0.5 * nc_cells. Thus, the voting rings
            #       are cropped.
            #    2. compared to the radius with max number of voted pixels, this
            #       radius has less than 50% of the votes.

            vote_fraction = np.count_nonzero(hough_res, axis=(1, 2)) / n_cells**2
            radius_mask = try_radii < 0.5 * n_cells
            votes_mask = vote_fraction > 0.50 * np.max(vote_fraction)
            mask = np.logical_and(radius_mask, votes_mask)
            if not np.any(mask):
                # return if there's no radii left
                if return_pixels_and_votes:
                    return None, None, None, None
                else:
                    return None
            hough_res = hough_res[mask]
            try_radii = try_radii[mask]

            hough_flattened = hough_res.reshape(hough_res.shape[0], -1)
            top_10 = np.partition(hough_flattened, -10, axis=1)[:, -10:]
            # discard radii where there's fewer than 10 candidates
            # i.e. where top 10 contains 0
            discard_mask = top_10.min(axis=1) < 1e-3
            top_10_normalized = top_10 / np.sum(top_10, axis=1, keepdims=True)
            top_10_entropy = -np.sum(
                top_10_normalized * np.log(top_10_normalized + 1e-12), axis=1
            )
            # replace NaNs with max value
            top_10_entropy[np.isnan(top_10_entropy)] = -1
            top_10_entropy[top_10_entropy < 0] = top_10_entropy.max()
            top_10_entropy[discard_mask] = top_10_entropy.max()
            # normalize entropy to be between 1/entropy_weighting and 1 given max reward
            # of entropy_weighting
            entropy_range = np.max(top_10_entropy) - np.min(top_10_entropy)
            if entropy_range < 1e-12:
                penalty = np.ones_like(top_10_entropy)
            else:
                penalty = (
                    1 / entropy_weighting
                    + (1 - 1 / entropy_weighting)
                    * (top_10_entropy - np.min(top_10_entropy))
                    / entropy_range
                )
            hough_res /= penalty[:, None, None]

        # constrain circles to be roughly above previous one
        if center_region is not None:
            previous_center = center_region.center[:2]
            search_radius = center_region.radius
            # calculate distance of every circle candidate center to the previous
            # center
            dist = np.sqrt(
                (grid_x - previous_center[0]) ** 2 + (grid_y - previous_center[1]) ** 2
            )
            # mask out all circles that are not within the search radius
            hough_res[np.broadcast_to(dist, hough_res.shape) > search_radius] = 0

        # find the circle with the most votes
        i_rad, x_px, y_px = np.unravel_index(np.argmax(hough_res), hough_res.shape)

        # transform pixel coordinates back to world coordinates
        x_c = grid_x[0, x_px, y_px] + grid_res / 2
        y_c = grid_y[0, x_px, y_px] + grid_res / 2
        r = try_radii[i_rad] * grid_res
        circ = cls((x_c, y_c, circle_height), r)

        if return_pixels_and_votes:
            return circ, pixels, hough_res[i_rad], penalty[i_rad, None, None]
        else:
            return circ

    @classmethod
    def from_cloud_bullock(
        cls,
        points: np.ndarray,
        weights: np.ndarray = None,
        fixed_radius: float = None,
        circle_height: float = 0.0,
        **kwargs,
    ) -> "Circle":
        """Fits a circle to a point cloud in the least-squares sense.

        Adapted from https://dtcenter.org/sites/default/files/community-code/met/docs/write-ups/circle_fit.pdf

        Args:
            points (np.ndarray): N x 2 array of points in the slice

        Returns:
            Circle: fitted circle
        """
        if fixed_radius is None:
            # normalize points
            if weights is None:
                weights = np.ones(points.shape[0])
            points_mean = np.mean(points, axis=0)
            u = points[:, 0] - points_mean[0]
            v = points[:, 1] - points_mean[1]
            # pre-calculate summands
            S_uu = np.sum(np.power(u, 2) * weights)
            S_vv = np.sum(np.power(v, 2) * weights)
            S_uv = np.sum(u * v * weights)
            S_uuu_uvv = np.sum((np.power(u, 3) + u * np.power(v, 2)) * weights)
            S_vvv_vuu = np.sum((np.power(v, 3) + v * np.power(u, 2)) * weights)

            # calculate circle center in normalized coordinates and radius
            v_c = (S_uuu_uvv / (2 * S_uu) - S_vvv_vuu / (2 * S_uv)) / (
                S_uv / S_uu - S_vv / S_uv + 1e-12
            )
            u_c = (S_uuu_uvv / (2 * S_uv) - S_vvv_vuu / (2 * S_vv)) / (
                S_uu / S_uv - S_uv / S_vv + 1e-12
            )
            r = np.sqrt(
                np.power(u_c, 2) + np.power(v_c, 2) + (S_uu + S_vv) / points.shape[0]
            )
            # denormalize
            x_c, y_c = points_mean[0] + u_c, points_mean[1] + v_c
            return cls((x_c, y_c, circle_height), r)
        else:
            r = fixed_radius
            A = np.hstack([2 * points, np.ones((points.shape[0], 1))])
            b = np.sum(np.power(points, 2), axis=1) - np.power(r, 2)
            if weights is not None:
                A = A * weights[:, None]
                b = b * weights
            c = np.linalg.lstsq(A, b, rcond=None)[0]
            x_c, y_c = c[:2]
            return cls((x_c, y_c, circle_height), r)

    @classmethod
    def from_cloud_ransac(
        cls,
        points: np.ndarray,
        min_radius: float = 0.0,
        max_radius: float = np.inf,
        num_samples: int = 500,
        center_region: "Circle" = None,
        min_n_inliers: int = 10,
        inlier_threshold: float = 0.01,
        circle_height: float = 0.0,
        sampling: str = "weighted",
        **kwargs,
    ) -> Tuple["Circle", np.ndarray]:
        """This function filters outliers using the RANSAC algorithm. It fits a
        circle to a random sample of n_points points and counts how many points are
        inliers. If the number of inliers is larger than min_n_inliers, the circle is
        accepted and the number of inliers is updated. The algorithm is repeated
        n_iterations times.

        Args:
            points (np.ndarray): N x 2 array of points in the slice
            n_iterations (int): Number of iterations to run the algorithm
            n_points (int): Number of points used to model the circle
            min_n_inliers (int): Minimum number of inliers to accept the circle
            inlier_threshold (float): thickness of band around circle to qualify
                                    inliers

        Returns:
            tuple: best model Circle and inlier points if model was found, else None
        """
        best_model = None
        points = points[:, :2]

        if sampling == "weighted":
            # construct weights by distance to closest neighbor i.e. local density
            dist_to_closest_neighbor = cKDTree(points).query(points, k=2)[0][:, 1]
            probas = np.exp(-dist_to_closest_neighbor)
            probas -= probas.min() - 1e-12
            probas /= probas.max()

        N_points = points.shape[0]
        N_circs = int(N_points * (N_points - 1) * (N_points - 2) * 0.01)
        N_circs = max(N_points, min(N_circs, num_samples))
        N_circs = min(max(N_points, N_circs), num_samples)

        if sampling == "weighted":
            indices = (
                (1 - np.random.rand(N_circs, len(points)) * probas)
            ).argpartition(3, axis=1)[:, :3]
        elif sampling == "random":
            indices = ((1 - np.random.rand(N_circs, len(points)))).argpartition(
                3, axis=1
            )[:, :3]
        elif sampling == "full":
            indices = np.array(list(product(range(len(points)), repeat=3)))
            mask = np.logical_and(
                indices[:, 0] != indices[:, 1],
                indices[:, 0] != indices[:, 2],
                indices[:, 1] != indices[:, 2],
            )
            mask = np.logical_and(mask, indices[:, 1] != indices[:, 2])
            indices = indices[mask]
        else:
            raise ValueError("sampling must be one of 'weighted', 'random', 'full'")

        circle_points = points[indices]  # num_samples x 3 x 2

        circle_points = points[indices]  # n_iters x 3 x 2
        circles = cls.from_3_2d_points(circle_points, method="bisectors")  # n_iters x 3

        circles = circles[circles[:, 2] < max_radius]
        circles = circles[circles[:, 2] > min_radius]
        if center_region is not None:
            dists = np.linalg.norm(circles[:, :2] - center_region.center[:2], axis=1)
            circles = circles[dists < center_region.radius]

        dists = np.linalg.norm(points[None, ...] - circles[:, None, :2], axis=2).T
        dists -= circles[:, 2][None, :]
        inlier_mask = np.logical_and(
            dists < inlier_threshold, dists > -inlier_threshold
        )
        n_inliers = np.sum(inlier_mask, axis=0)
        circles = circles[n_inliers > min_n_inliers]
        n_inliers = n_inliers[n_inliers > min_n_inliers]
        if len(circles) == 0:
            return None

        best_model = circles[np.argmax(n_inliers)]
        best_model = cls((best_model[0], best_model[1], circle_height), best_model[2])
        return best_model

    @classmethod
    def from_cloud_hough_ransac(
        cls,
        points,
        min_radius: float = 0.0,
        max_radius: float = np.inf,
        max_num_samples: int = 500,
        center_region: "Circle" = None,
        search_radius: float = 0.05,
        sample_percentage: float = 0.01,
        sampling: str = "weighted",
        circle_height: float = 0.0,
        **kwargs,
    ) -> "Circle":
        """Fits a circle to a point cloud using the hough space to select the best fit
        in a RANSAC sense. Reducing the range between min_radius and max_radius as well
        as providing a center_region can speed up the algorithm.

        Args:
            points (np.ndarray): Input point cloud (N x 2)
            min_radius (float, optional): lower bound for circle. Defaults to 0.0.
            max_radius (float, optional): upper bound for circle. Defaults to np.inf.
            max_num_samples (int, optional): number of samples to search the hough space.
                Defaults to 500.
            center_region (Circle, optional): initialization region for circle.
                Defaults to None.
            search_radius (float, optional): Search radius of sphere in hough space to
                find the point with highest density. Defaults to 0.05.
            sample_percentage (float, optional): Percentage of all possible triplets to
                be sampled. 1 percent is a good value. This is bounded by the number of
                input points (lb) and max_num_samples (ub). Defaults to 0.01.
            sampling (str, optional): Method of sampling triplets. "random" treats all
                points equally, with "weighted", points with a close neighbor
                (i.e. high local density) are preferred, and "full" ignores all bounds
                and uses all possible triplets. Defaults to "weighted".
            circle_height (float, optional): height for fitted circle. Defaults to 0.0.

        Raises:
            ValueError: wrong sampling method

        Returns:
            Circle: the fitted circle
        """
        if len(points) < 10:
            return None
        points = points[:, :2]
        # sampling triplets of points
        if sampling == "weighted":
            # construct weights by distance to closest neighbor i.e. local density
            dist_to_closest_neighbor = cKDTree(points).query(points, k=2)[0][:, 1]
            probas = np.exp(-dist_to_closest_neighbor)
            probas -= probas.min() - 1e-12
            probas /= probas.max()

        N_points = points.shape[0]
        N_circs = int(N_points * (N_points - 1) * (N_points - 2) * sample_percentage)
        N_circs = min(max(N_points, N_circs), max_num_samples)

        if sampling == "weighted":
            indices = (
                (1 - np.random.rand(N_circs, len(points)) * probas)
            ).argpartition(3, axis=1)[:, :3]
        elif sampling == "random":
            indices = ((1 - np.random.rand(N_circs, len(points)))).argpartition(
                3, axis=1
            )[:, :3]
        elif sampling == "full":
            indices = np.array(list(product(range(len(points)), repeat=3)))
            mask = np.logical_and(
                indices[:, 0] != indices[:, 1],
                indices[:, 0] != indices[:, 2],
                indices[:, 1] != indices[:, 2],
            )
            mask = np.logical_and(mask, indices[:, 1] != indices[:, 2])
            indices = indices[mask]
        else:
            raise ValueError("sampling must be one of 'weighted', 'random', 'full'")

        circle_points = points[indices]  # num_samples x 3 x 2

        # fitting circles to triplets
        circles = cls.from_3_2d_points(circle_points, method="bisectors")

        # filter circles using constraints
        circles = circles[circles[:, 2] < max_radius]
        circles = circles[circles[:, 2] > min_radius]
        if center_region is not None:
            dists = np.linalg.norm(circles[:, :2] - center_region.center[:2], axis=1)
            circles = circles[dists < center_region.radius]

        if len(circles) == 0:
            return None

        query_circles = circles.copy()
        query_circles[:, 2] *= 2
        kdtree = cKDTree(query_circles)
        neighbors = kdtree.query_ball_tree(kdtree, r=search_radius)
        best_index = neighbors.index(max(neighbors, key=len))
        best_circle_inds = neighbors[best_index] + [best_index]
        best_circles = circles[best_circle_inds]

        # best_circle = np.average(best_circles, weights=probas[best_circle_inds], axis=0)
        best_circle = best_circles.mean(axis=0)

        return cls(
            center=(best_circle[0], best_circle[1], circle_height),
            radius=best_circle[2],
        )

    @classmethod
    def from_3_2d_points(
        cls, points: np.ndarray, method: str = "bisectors"
    ) -> Union["Circle", np.ndarray]:
        """Finds the circle that intersects three points in 2D.

        Args:
            points (np.ndarray): input points. Either 1 set of points (3x2), then a
                Circle is returned, or several circles (N, 3, 2).
            method (str, optional): Either "bisectors" (fastest), or
                "determinant" (textbook). Defaults to "bisectors".

        Raises:
            ValueError: method not implemented or wrong shape of points

        Returns:
            "Circle" of np.ndarray: depending of the input, a circle object or a
                np.ndarray (Nx3) with the columns being x, y, r
        """
        # from https://qc.edu.hk/math/Advanced%20Level/circle%20given%203%20points.htm
        if points.shape[0] != 3:
            assert points.shape[1] == 3, "each batch must contain 3 points"
        else:
            assert points.shape[0] == 3, "points must contain 3 points"
            points = points[None, ...]
        if method == "bisectors":
            # intersecting bisectors method
            p0 = (points[:, 0, :] + points[:, 1, :]) / 2
            p1 = (points[:, 1, :] + points[:, 2, :]) / 2
            v0 = points[:, 1, :] - points[:, 0, :]
            v0 = np.flip(v0, axis=1) * np.array([1, -1])
            v1 = points[:, 2, :] - points[:, 1, :]
            v1 = np.flip(v1, axis=1) * np.array([1, -1])
            with np.errstate(divide="ignore", invalid="ignore"):
                alpha = (
                    p1[:, 1] - p0[:, 1] + (p0[:, 0] - p1[:, 0]) * v1[:, 1] / v1[:, 0]
                )
                alpha /= v0[:, 1] - v0[:, 0] * v1[:, 1] / v1[:, 0]
            x = p0[:, 0] + alpha * v0[:, 0]
            y = p0[:, 1] + alpha * v0[:, 1]
            r = np.sqrt((x - points[:, 0, 0]) ** 2 + (y - points[:, 0, 1]) ** 2)
        elif method == "determinant":
            # determinant method
            M = np.zeros((points.shape[0], 3, 4))
            M[:, 0, 0] = points[:, 0, 0] ** 2 + points[:, 0, 1] ** 2
            M[:, 0, 1] = points[:, 0, 0]
            M[:, 0, 2] = points[:, 0, 1]
            M[:, 0, 3] = 1
            M[:, 1, 0] = points[:, 1, 0] ** 2 + points[:, 1, 1] ** 2
            M[:, 1, 1] = points[:, 1, 0]
            M[:, 1, 2] = points[:, 1, 1]
            M[:, 1, 3] = 1
            M[:, 2, 0] = points[:, 2, 0] ** 2 + points[:, 2, 1] ** 2
            M[:, 2, 1] = points[:, 2, 0]
            M[:, 2, 2] = points[:, 2, 1]
            M[:, 2, 3] = 1
            denom = np.linalg.det(M[:, :, 1:])
            x = 0.5 * np.linalg.det(M[:, :, [0, 2, 3]]) / denom
            y = -0.5 * np.linalg.det(M[:, :, [0, 1, 3]]) / denom
            r = np.sqrt(x**2 + y**2 + np.linalg.det(M[:, :, :3]) / denom)
        else:
            raise ValueError(f"method {method} not supported")
        circles = np.stack([x, y, r], axis=1)
        if points.shape[0] == 1:
            return cls((circles[0, 0], circles[0, 1], 0), circles[0, 2])
        else:
            return circles

    def query_point(self, theta: float) -> np.ndarray:
        """returns a point at a given angle on the circle

        Args:
            theta (float): angle of evaluation towards x-axis

        Returns:
            np.ndarray: Point on circle
        """
        pointer = np.array([np.cos(theta), -np.sin(theta), 0])
        return self.center + self.radius * self.rot_mat @ pointer

    def get_distance(self, point: np.ndarray, use_z: bool = False) -> np.ndarray:
        """Evaluates the distance of a point to the circle

        Args:
            point (np.ndarray): point(s) to evaluate
            use_z (bool, optional): If the z-component should be considere.
                Defaults to False.

        Returns:
            np.ndarray: distance of point(s) to circle
        """
        if use_z:
            if len(point.shape) == 2:
                return np.linalg.norm(point - self.center, axis=1) - self.radius
            else:
                return np.linalg.norm(point - self.center) - self.radius
        else:
            if len(point.shape) == 2:
                return (
                    np.linalg.norm(point[:, :2] - self.center[:2], axis=1) - self.radius
                )
            else:
                return np.linalg.norm(point[:2] - self.center[:2]) - self.radius

    def genereate_cone_frustum_mesh(
        self, other_circle: "Circle", num_verts: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """This function genereates meshes (vertices and triangle indices) for a frustum
        (cut cone-oid) spanned by the current circle and a given other circle.

        Args:
            other_circle (Circle): Other circle spannig the frustum
            num_verts (int, optional): Number of vertices for vizualizing the top or
                bottom face. Defaults to 100.

        Returns:
            np.ndarray, np.ndarray: vertices and triangles
        """
        vertices = [
            self.query_point(theta) for theta in np.linspace(0, 2 * np.pi, num_verts)
        ] + [
            other_circle.query_point(theta)
            for theta in np.linspace(0, 2 * np.pi, num_verts)
        ]
        vertices = np.stack(vertices)
        basic_tri_1 = np.array([0, num_verts, 1])  # these triangles repeats 100 times
        basic_tri_2 = np.array([1, num_verts, num_verts + 1])
        tri_indices = [basic_tri_1 + i for i in range(num_verts)] + [
            basic_tri_2 + i for i in range(num_verts)
        ]
        tri_indices = np.stack(tri_indices)[:-1, :]  # last is equal to first

        return vertices, tri_indices

    def __str__(self) -> str:
        """generates a string representation of the circle object

        Returns:
            str: String representation
        """
        return f"Circle: center: {self.center}, radius: {self.radius}, normal: {self.rot_mat[:, 2]}"

    def apply_transform(self, translation: np.ndarray, rotation: np.ndarray):
        """Applies a transform to the circle

        Args:
            translation (np.ndarray): translation vector
            rotation (np.ndarray): either a 3x3 rotation matrix or a quaternion

        Raises:
            ValueError: if the rotation format is wrong
        """
        if rotation.shape[0] == 4:
            rot_mat = Rotation.from_quat(rotation).as_matrix()
        elif rotation.shape == (3, 3):
            rot_mat = rotation
        else:
            raise ValueError("rotation must be given as 3x3 matrix or quaternion")
        self.center = rot_mat @ self.center + translation
        self.rot_mat = rot_mat @ self.rot_mat
