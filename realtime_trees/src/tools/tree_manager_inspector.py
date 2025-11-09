#!/usr/bin/env python3

from colorsys import hls_to_rgb
import numpy as np
import open3d as o3d
import open3d.visualization as vis

from realtime_trees.tree_manager import TreeManager


# tm = TreeManager.from_zip("tm.zip")
tm = TreeManager.from_zip("realtime_trees_ros/output/trees/logs/raw/tree_manager_conifer_2.zip")


def create_scene():
    geoms = []
    for tree in tm.trees:
        hue = np.random.rand()
        point_cloud = o3d.geometry.PointCloud()
        for cluster in tree.clusters:
            pc = cluster.cloud.to_legacy()
            pc.paint_uniform_color(hls_to_rgb(hue, np.random.rand(), 1))
            pc.transform(cluster.info.T_sensor2map)
            point_cloud += pc
        geoms.append({
        "name": f"Tree_{tree.id:0>3}",
        "geometry": point_cloud
    })
    ids = np.array([tree.id for tree in tm.trees])
    args_coverage = np.argsort(tm.tree_coverage_angles)[::-1]
    args_num_clusters = np.argsort([len(tree.clusters) for tree in tm.trees])[::-1]
    print(f"Trees with biggest coverage angles: {args_coverage[:10]}")
    print(f"Tree with the most clusters: {ids[args_num_clusters][:10]}")
    return geoms


def on_init(o3dvis: vis.O3DVisualizer):
    o3dvis.ground_plane = o3d.visualization.rendering.Scene.GroundPlane.XY
    for tree in tm.trees:
        o3dvis.show_geometry(f"Tree_{tree.id:0>3}", show=False)
    pass

if __name__ == "__main__":
    geoms = create_scene()
    vis.draw(geoms,
             bg_color=(0.8, 0.9, 0.9, 1.0),
             show_ui=True,
             width=1920,
             height=1080,
             on_init=on_init,)