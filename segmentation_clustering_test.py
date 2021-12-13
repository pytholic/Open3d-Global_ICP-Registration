import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Utility Functions
def load_mesh(path):
    mesh = o3d.io.read_triangle_mesh(path, enable_post_processing=True)
    return mesh

def mesh_to_array(data):
    """
    Input:
        Object file
    Ouput:
        Nx3 array
    """

    xyz = np.asarray(data.vertices, dtype=np.float32)
    return xyz

def array_to_pcd(data):
    """
    Input:
        Nx3 array
    """

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    return pcd

obj = load_mesh("./data_aumc/3/scene.obj") # 30, 1, 3
xyz = mesh_to_array(obj)
pcd = array_to_pcd(xyz) # target

pcd.paint_uniform_color([0.5, 0.5, 0.5])
o3d.visualization.draw_geometries([pcd])


plane_model, inliers = pcd.segment_plane(distance_threshold=0.025,
                                         ransac_n=3,
                                         num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
#print(type(inliers))

inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1, 0.67, 0])
outlier_cloud = pcd.select_by_index(inliers, invert=True)
outlier_cloud.paint_uniform_color([0, 0.651, 0.929])
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

o3d.visualization.draw_geometries([outlier_cloud])

with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(outlier_cloud.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
outlier_cloud.colors = o3d.utility.Vector3dVector(colors[:,:3])
o3d.visualization.draw_geometries([outlier_cloud])

# print(outlier_cloud)
print(len(labels))
# print(type(labels))

# print(np.bincount(labels).argmax())
# print(Counter(labels).most_common(1)[0][0])
# print(Counter(labels).most_common(1))

idx_freq = Counter(labels).most_common(1)[0][0]
#idx_array = labels[labels == idx_freq]
idx_array = np.where(labels == idx_freq)
print(idx_array[0])
print(len(idx_array[0]))

final_pcd = outlier_cloud.select_by_index(list(idx_array[0]))
print(final_pcd)

final_pcd.paint_uniform_color([0.5, 0.5, 0.5])
o3d.visualization.draw_geometries([final_pcd])