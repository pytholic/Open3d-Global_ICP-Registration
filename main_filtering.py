import open3d as o3d
import numpy as np
import copy
import time
from segmentation_clustering import *
from utils import *


# Helper visualization function
def draw_registration_result(source, target, transformation):
    """
    This helper function `visualizes` the transformed 
    source point cloud together with the target 
    point cloud.
    """

    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

# Extract geometric features
def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

# Read point clouds and misalign them
def prepare_dataset(voxel_size, source, target):
    print(":: Load two point clouds and disturb initial pose.")
    
    # Misalign with a rotation matrix as transformation
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], # rotate around y-axis by 90 degrees
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

# Run global registration
def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1  # 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

# Run ICP registration
def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return result


def registration(source, target, is_filter=True):
    """
    source: model point cloud
    target: scene point cloud
    filter: flag for filter function
    """

    if is_filter == True:
        inlier_cloud, outlier_cloud = plane_segmentation(target)
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

        pcd_clustered, labels = clustering(outlier_cloud)
        o3d.visualization.draw_geometries([pcd_clustered])

        idx_freq = Counter(labels).most_common(1)[0][0]
        idx_array = np.where(labels == idx_freq)

        target_filtered = pcd_clustered.select_by_index(list(idx_array[0]))
        target_filtered.paint_uniform_color([0, 0.651, 0.929])
        o3d.visualization.draw_geometries([target_filtered])

        voxel_size = 0.007   # means 0.5cm for this dataset => 0.5 cm = 1 voxel
        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size, source=source, target=target_filtered)

        start = time.time()
        result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
        print("Global registration took %.3f sec.\n" % (time.time() - start))
        print(result_ransac)
        draw_registration_result(source_down, target_down, result_ransac.transformation)


        result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
                                     voxel_size, result_ransac)
        print(result_icp)
        draw_registration_result(source, target, result_icp.transformation)

    else:
        voxel_size = 0.005  # means 0.5cm for this dataset => 0.5 cm = 1 voxel
        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size, source=source, target=target)

        start = time.time()
        result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
        print("Global registration took %.3f sec.\n" % (time.time() - start))
        print(result_ransac)
        draw_registration_result(source_down, target_down, result_ransac.transformation)


        result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
                                     voxel_size, result_ransac)
        print(result_icp)
        draw_registration_result(source, target, result_icp.transformation)


if __name__=="__main__":

    source_obj = load_mesh("./data_aumc/1/model.obj")
    source_xyz = mesh_to_array(source_obj)
    source = array_to_pcd(source_xyz) # source

    target_obj = load_mesh("./data_aumc/1/scene.obj") # 30, 1, 3
    target_xyz = mesh_to_array(target_obj)
    target = array_to_pcd(target_xyz) # target

    source.paint_uniform_color([1, 0.706, 0])
    target.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([source, target])

    registration(source, target, is_filter=True)

    
