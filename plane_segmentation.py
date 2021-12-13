import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

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

obj = load_mesh("./data_aumc/2/scene.obj")
xyz = mesh_to_array(obj)
pcd = array_to_pcd(xyz) # target

pcd.paint_uniform_color([0.5, 0.5, 0.5])
o3d.visualization.draw_geometries([pcd])


plane_model, inliers = pcd.segment_plane(distance_threshold=0.025,
                                         ransac_n=3,
                                         num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1, 0.67, 0])
outlier_cloud = pcd.select_by_index(inliers, invert=True)
outlier_cloud.paint_uniform_color([0, 0.651, 0.929])
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

o3d.visualization.draw_geometries([outlier_cloud])