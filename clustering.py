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

obj = load_mesh("./data_aumc/1/scene.obj")
xyz = mesh_to_array(obj)
pcd = array_to_pcd(xyz) # target

pcd.paint_uniform_color([1, 0.706, 0])
o3d.visualization.draw_geometries([pcd])

with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(pcd.cluster_dbscan(eps=0.005, min_points=10, print_progress=True))

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:,:3])
o3d.visualization.draw_geometries([pcd])