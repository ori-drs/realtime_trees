from dataclasses import dataclass, field
from typing import Tuple, Optional
import numpy as np
import trimesh
import open3d as o3d

from realtime_trees.utils.ros_proxys import Time

@dataclass
class ClusterAxis:
    radius: float
    score: float
    height: float = None
    transform: np.ndarray = field(default_factory=lambda: np.eye(4))  # 4x4 transform matrix

@dataclass
class CoverageInfo:
    angle_from: float
    angle_to: float
    distance_min: float
    distance_max: float

@dataclass  
class ClusterInfo:
    id: int
    color: Tuple[float, float, float] = (0, 0, 0)
    axis: Optional[ClusterAxis] = None
    T_sensor2map: Optional[np.ndarray] = field(default_factory=lambda: np.eye(4))  # sensor to map transform
    coverage: Optional[CoverageInfo] = None  # coverage info
    time_stamp: Optional[Time] = None  # time stamp of the observation

@dataclass
class Cluster:
    cloud: o3d.t.geometry.PointCloud
    info: ClusterInfo
    
@dataclass
class Terrain:
    mesh_sensor: trimesh.Trimesh
    vertex_weights: np.ndarray
    path: np.ndarray
    cell_size: float
    time_stamp: Time
    T_sensor2map: np.ndarray = field(default_factory=lambda: np.eye(4))
    