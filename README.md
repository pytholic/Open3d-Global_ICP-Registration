# Open3d-Global_ICP-Registration
Global registration plus ICP refinement in open3d

## Global Registration
Both **ICP registration** and **Colored point cloud registration** are known as local registration methods because they rely on a rough alignment as initialization. This tutorial shows another class of registration methods, known as **global registration**. This family of algorithms do not require an alignment for initialization. They usually produce less tight alignment results and are used as initialization of the local methods.

### Visualization
This helper function `visualizes` the transformed source point cloud together with the target point cloud.

### Extract Geometric Feature
We downsample the point cloud, estimate normals, then compute a **FPFH** feature for each point. The FPFH feature is a 33-dimensional vector that describes the local geometric property of a point. A nearest neighbor query in the 33-dimensional space can return points with similar local geometric structures.

### Input
We read a source point cloud and a target point cloud from two files. They are misaligned with an `identity matrix` as transformation.

### RANSAC
RANSAC (Random Sample Consensus) is used to deal with the `outliers` in the data associations or identifying which points are `inliers` and `outliers` for our model estimation technique.

We use **RANSAC** for global registration. In each RANSAC iteration, `ransac_n` random points are picked from the source point cloud. Their corresponding points in the target point cloud are detected by querying the nearest neighbor in the 33-dimensional FPFH feature space. A pruning step takes fast pruning algorithms to quickly reject false matches early. Open3D provides the following pruning algorithms:

* **CorrespondenceCheckerBasedOnDistance** checks if aligned point clouds are close i.e. less than the specified `threshold`.
* **CorrespondenceCheckerBasedOnEdgeLength** checks if the lengths of any two arbitrary edges (line formed by two vertices) individually drawn from source and target correspondences are similar. This example checks that `||edgesource||>0.9⋅||edgetarget||` and `||edgetarget||>0.9⋅||edgesource||` are true.
* **CorrespondenceCheckerBasedOnNormal** considers vertex normal affinity of any correspondences. It computes the `dot product` of two normal vectors. It takes a radian value for the threshold.

Only matches that pass the **pruning step** are used to compute a `transformation`, which is validated on the entire point cloud. The core function is `registration_ransac_based_on_feature_matching`. The most important hyperparameter of this function is `RANSACConvergenceCriteria`. It defines the maximum number of RANSAC iterations and the confidence probability. The larger these two numbers are, the more accurate the result is, but also the more time the algorithm takes.

We set the RANSAC parameters based on the empirical value provided by **Choi2015**.

### RANSAC Algorithm
Explanation by [Cyrill Stachniss](https://www.youtube.com/watch?v=Cu1f6vpEilg&t=251s)

**Steps**
1. **Sample** the number of data points required to fit the model
2. **Compute** the model parameters using the sampled data points
3. **Score** by the fraction of `inliers` within a preset `threshold` of the model

Repeat steps 1-3 until the best model is found with high confidence.

**How often do we need to try?**
Not easy to answer. Let's see which `parameters` are involved in our RANSAC estimation and how the affect the **number of trials** that we need to do.

* Number of sampeld points **s** (minimum number needed to fit the model)
* Outlier ratio **e** (e = #outliers / #datapoints)

What is requried Number of trials **T**?
Choose **T** so that, with probability **p**, at least one random sample set is free 
from outliers. 

## Local Refinement
For performance reason, the global registration is only performed on a heavily `down-sampled` point cloud. The result is also not tight. We use **Point-to-plane ICP** to further refine the alignment.

## Link
Example [link](http://www.open3d.org/docs/0.12.0/tutorial/pipelines/global_registration.html) has been provided by **Open3d** in the doumentation.

## Additional Observations
An implementation for **Fast Global Registration** has also been provided in the official example. However, it seems to be slower comapred to normal implementation due to some reason. Also, simple global registration is already fast enoug hfor our case.