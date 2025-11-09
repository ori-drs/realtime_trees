import CSF
import numpy as np
import open3d as o3d


def fit_terrain(
    cloud: o3d.t.geometry.PointCloud,
    sloop_smooth: bool = False,
    cloth_cell_size: float = 1.0,
) -> np.ndarray:
    """Fits a terrain to a point cloud using a cloth simulation filter.

    Args:
        cloud (o3d.t.geometry.PointCloud): input point cloud
        sloop_smooth (bool, optional): CSF parameter to apply smoothing post processor.
            Defaults to False.
        cloth_cell_size (float, optional): Cell size of cloth. Defaults to 1.0.

    Returns:
        np.ndarray: cloth in form of a mesh grid (MxNx3) where cloth[i, j] = [x, y, z]
    """
    csf = CSF.CSF()
    csf.params.bSloopSmooth = sloop_smooth
    csf.params.cloth_resolution = cloth_cell_size

    cloud = cloud.voxel_down_sample(voxel_size=csf.params.cloth_resolution / 4)

    csf.setPointCloud(cloud.point.positions.numpy().tolist())
    csf_mesh = csf.do_cloth_export()
    verts = np.array(csf_mesh).reshape((-1, 3))

    verts[:, :2] = verts[:, :2].round(decimals=3)
    num_x_cos = np.unique(verts[:, 0]).shape[0]
    num_y_cos = np.unique(verts[:, 1]).shape[0]
    cloth = verts.reshape((num_y_cos, num_x_cos, 3)).transpose((1, 0, 2))
    return cloth