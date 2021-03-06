# Open3d-Global_ICP-Registration
Global registration plus ICP refinement in open3d

## Data
My data consists of patient head scans (wavefront .obj format). Each patient has two mesh models i.e one is `scene` mesh which is obtained from a 3D structure sensor, and the other one is `model` mesh which is obtained from CT data.

## Filtering
Before we proceed to the **Global Registration** part, we might need to look at the filtering part. The reason for that is that sometimes the point cloud data can have a lot of outlier points i.e background parts which can also be termed as the `noise` points. 

The presence of such outliers can result in the bad performance of the global registration algorithm. Therefore it is important to remove those unwanted points from our data so that we can get better and more robust registration results.

In my approach, I used two methods from [**Open3D**](http://www.open3d.org/) library to filter my data. Those methods are `Plane Segmentation` and `Clustering`. Some documentation on these methods can be found at this [link](http://www.open3d.org/docs/latest/tutorial/Basic/pointcloud.html).

In my case, `model` object did not have much noise so it can be used as it is. However, the `scene` objects have a lot of background noise.

<!---![Scene Noise](./images/scene_noise.png)--->

<p align="center">
  <img width="400" height="380" src="./images/scene_noise.png">
</p>

In first step, I perform **plane segmentation** in Open3D and obtain the following separation between inlier and outlier points.

<p align="center">
  <img width="400" height="380" src="./images/plane_segmentation.png">
</p>

After plane segmentation, we still have some outlier parts in the point cloud. To remove those outliers, we can leverage **clustering** method provided in Open3D.

<p align="center">
  <img width="400" height="380" src="./images/clustering.png">
</p>

After **filtering**, the final point cloud looks like this.

<p align="center">
  <img width="400" height="380" src="./images/final.png">
</p>

**Note:** One thing to mention is that this filtering technique is applicable if the point cloud has **a lot of background points**. If your data has less or no noise and you apply these techniques, it can remove important parts from your data and hence can result in wrong registration. So we need to be careful and selective as to when we should apply filtering methods. 

## Global Registration
Both **ICP registration** and **Colored point cloud registration** are known as local registration methods because they rely on a rough alignment as initialization. This tutorial shows another class of registration methods, known as **global registration**. This family of algorithms do not require an alignment for initialization. They usually produce less tight alignment results and are used as initialization of the local methods.

### Visualization
This helper function `visualizes` the transformed source point cloud together with the target point cloud.

### Extract Geometric Feature
We downsample the point cloud, estimate normals, then compute a **FPFH** feature for each point. The FPFH feature is a 33-dimensional vector that describes the local geometric property of a point. A nearest neighbor query in the 33-dimensional space can return points with similar local geometric structures.

### Input
We read a source point cloud and a target point cloud from two files. They are misaligned with a `rotation matrix` as transformation.

### RANSAC
RANSAC (Random Sample Consensus) is used to deal with the `outliers` in the data associations or identifying which points are `inliers` and `outliers` for our model estimation technique.

We use **RANSAC** for global registration. In each RANSAC iteration, `ransac_n` random points are picked from the source point cloud. Their corresponding points in the target point cloud are detected by querying the nearest neighbor in the 33-dimensional FPFH feature space. A pruning step takes fast pruning algorithms to quickly reject false matches early. Open3D provides the following pruning algorithms:

* **CorrespondenceCheckerBasedOnDistance** checks if aligned point clouds are close i.e. less than the specified `threshold`.
* **CorrespondenceCheckerBasedOnEdgeLength** checks if the lengths of any two arbitrary edges (line formed by two vertices) individually drawn from source and target correspondences are similar. This example checks that `||edgesource||>0.9???||edgetarget||` and `||edgetarget||>0.9???||edgesource||` are true.
* **CorrespondenceCheckerBasedOnNormal** considers vertex normal affinity of any correspondences. It computes the `dot product` of two normal vectors. It takes a radian value for the threshold.

Only matches that pass the **pruning step** are used to compute a `transformation`, which is validated on the entire point cloud. The core function is `registration_ransac_based_on_feature_matching`. The most important hyperparameter of this function is `RANSACConvergenceCriteria`. It defines the maximum number of RANSAC iterations and the confidence probability. The larger these two numbers are, the more accurate the result is, but also the more time the algorithm takes.

We set the RANSAC parameters based on the empirical value provided by **Choi2015**.

### RANSAC Algorithm
Explanation by [Cyrill Stachniss](https://www.youtube.com/watch?v=Cu1f6vpEilg&t=251s).

<br />

**Steps**
1. **Sample** the number of data points required to fit the model
2. **Compute** the model parameters using the sampled data points
3. **Score** by the fraction of `inliers` within a preset `threshold` of the model

Repeat steps 1-3 until the best model is found with high confidence.

<br />

**How often do we need to try?**

Not easy to answer. Let's see which `parameters` are involved in our RANSAC estimation and how the affect the **number of trials** that we need to do.

* Number of sampled points **s** (minimum number needed to fit the model). Only inlier points are included.
* Outlier ratio **e** (e = #outliers / #datapoints)

<br />

**What is requried Number of trials T?**

Choose **T** so that, with probability **p**, at least one random sample set is free 
from outliers.

For one `iteration`, how likely is it to to succeed and find the right result? What does it mean to find the right result?

To find the right result mean sampling with **s** containing only the `inlier` points and there is not a single `outlier` in this set. The nwe can comput the `model` only using the inliers, and then have correct results.

<br />

**What is the probability of only drawing inliers?**

The probability of drawing a single inlier is `1-e`. Therefore the probability for drawing all the inliers is `(1-e)(1-e)(1-e)...`, i.e. `p = (1-e)^s`. 

<br />

**What is the probability of failing?**

Failing means that one of those points is an oulier. So it will be `1 - p = 1 - (1-e)^s`. This is the probability of failing **once** in one iteration.

Then what is the probability of failing **T** times? i.e. What is the probability of failing always?\
It will be `1 - p = 1-((1-e)^s)^T`.

**T** is the only unknown here. We can rearrange the equation and compute **T** based on **p**, **e** and **s**. We can take `log` on both sides and get the following equation.

`log(1-p) = Tlog(1 - (1-e)^s)`

`T = log(1-p) / log(1 - (1-e)^s)`

This gives us the numeber of trials that we need.

## Local Refinement
For performance reason, the global registration is only performed on a heavily `down-sampled` point cloud. The result is also not tight. We use **Point-to-plane ICP** to further refine the alignment.

## Link
Example [link](http://www.open3d.org/docs/0.12.0/tutorial/pipelines/global_registration.html) has been provided by **Open3d** in the doumentation.

## Additional Observations
An implementation for **Fast Global Registration** has also been provided in the official example. However, it seems to be slower comapred to normal implementation due to some reason. Also, simple global registration is already fast enoug hfor our case.