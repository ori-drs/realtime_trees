import numpy as np
from pymap3d import enu2geodetic
from scipy.spatial.transform import Rotation as R


def map2lla(pos, lla_ref, lla_r2m):
    assert len(lla_r2m) == 7, \
        "transform from reference to map must be 7 numbers (3 trans, 4 quat)"
    R_map_to_lla = R.from_quat(lla_r2m[3:]).as_matrix()
    t_map_to_lla = -1.0 * np.array(lla_r2m[:3])
    pos_enu = np.dot(R_map_to_lla, pos) + t_map_to_lla
    return enu2geodetic(pos_enu[0], pos_enu[1], pos_enu[2], *lla_ref)