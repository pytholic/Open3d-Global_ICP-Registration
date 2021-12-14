import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from utils import *


def plane_segmentation(pcd):
    """
    Input: Point Cloud Data
    """

    plane_model, inliers = pcd.segment_plane(distance_threshold=0.025,
                                         ransac_n=3,
                                         num_iterations=1000)
    [a, b, c, d] = plane_model
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1, 0.67, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    outlier_cloud.paint_uniform_color([0, 0.651, 0.929])
    return inlier_cloud, outlier_cloud
    #o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

def clustering(pcd):
    """
    Input: Point Cloud Data
    """

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:,:3])
    return pcd, labels
    #o3d.visualization.draw_geometries([pcd])


if __name__=="__main__":


    obj = load_mesh("./data_aumc/3/scene.obj") # 30, 1, 3
    xyz = mesh_to_array(obj)
    pcd = array_to_pcd(xyz) # target

    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.visualization.draw_geometries([pcd])

    inlier_cloud, outlier_cloud = plane_segmentation(pcd)
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    pcd_clustered, labels = clustering(outlier_cloud)
    o3d.visualization.draw_geometries([pcd_clustered])

    idx_freq = Counter(labels).most_common(1)[0][0]
    idx_array = np.where(labels == idx_freq)

    final_pcd = pcd_clustered.select_by_index(list(idx_array[0]))

    final_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.visualization.draw_geometries([final_pcd])