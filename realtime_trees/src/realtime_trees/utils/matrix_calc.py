import numpy as np
from scipy.spatial.transform import Rotation


def efficient_inv(T: np.ndarray) -> np.ndarray:
    """Efficient method to invert an affine transformation matrix

    Args:
        T (np.ndarray): matrix to invert 

    Returns:
        np.ndarray: inverted matrix
    """
    assert T.shape == (4, 4), "T must be a 4x4 matrix"
    T_inv = np.eye(4)
    T_inv[:3, :3] = T[:3, :3].T
    T_inv[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return T_inv

def apply_transform(
    points: np.ndarray,
    translation: np.ndarray,
    rotation: np.ndarray,
    inverse: bool = False,
) -> np.ndarray:
    """Transforms the given points by the given translation and rotation.
    Optionally performs the inverse transformation.

    Args:
        points (np.ndarray): Nx3 array of points to be transformed
        translation (np.ndarray): 3x1 array of translation
        rotation (np.ndarray): 3x3 rotation matrix or 4x1 quaternion
        inverse (bool, optional): Flag to calculate inverse. Defaults to False.

    Returns:
        np.ndarray: Nx3 array of transformed points
    """
    points = points.copy()
    if rotation.shape[0] == 4:
        rot_mat = Rotation.from_quat(rotation).as_matrix()
    elif rotation.shape == (3, 3):
        rot_mat = rotation
    else:
        raise ValueError("rotation must be given as 3x3 matrix or quaternion")
    if inverse:
        points = (points - translation) @ rot_mat
    else:
        points = points @ rot_mat.T + translation

    return points

def T_to_pos_and_quat(T: np.ndarray) -> tuple:
    """Extracts the position and quaternion from a 4x4 transformation matrix

    Args:
        T (np.ndarray): 4x4 transformation matrix

    Returns:
        tuple: position (3x1) and quaternion (4x1)
    """
    assert T.shape == (4, 4), "T must be a 4x4 matrix"
    return T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat()