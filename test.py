import open3d as o3d
import numpy as np


mesh = o3d.io.read_triangle_mesh("./data_aumc/48/model_3.obj", enable_post_processing=True)
o3d.visualization.draw_geometries([mesh])