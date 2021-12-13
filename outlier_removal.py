import open3d as o3d
import numpy as np

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

obj = load_mesh("./data_aumc/1/scene.obj")
xyz = mesh_to_array(obj)
pcd = array_to_pcd(xyz) # target

pcd.paint_uniform_color([1, 0.706, 0])
o3d.visualization.draw_geometries([pcd])

# print("Downsample the point cloud with a voxel of 0.02")
# voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)

# voxel_down_pcd.paint_uniform_color([1, 0.706, 0])
# o3d.visualization.draw_geometries([voxel_down_pcd])


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


print("Statistical oulier removal")
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=2.0)
display_inlier_outlier(pcd, ind)


print("Radius oulier removal")
cl, ind = pcd.remove_radius_outlier(nb_points=500, radius=0.05)
display_inlier_outlier(pcd, ind)