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