---
layout: post
title: 3D Deep Learning
abstract: A short text explaining what this post is about.
tags: [tag1, tag2, tag3]
category: category
mathjax: true
update: 0000-01-01 
time: 0
words: 0
---

# {{ page.title }}

While working with the robots at DLR (the German aerospace centre), I've been confronted with a new type of data---next to camera images---which I haven't come across so far, namely _point clouds_. As it turns out, point clouds can be an extremely useful extension to the two dimensional RGB camera images already commonly used in scene analysis, for example for object recognition and classification.

However, there are differences between the data types which prevent us from directly applying successful techniques in one area to another. In this post, I'd like to explore those properties after a detailed look at point clouds themselves, to then see which ideas have been employed to extend the deep learning revolution to this promising data type.

## What exactly _is_ a point cloud?

As the name suggests, a point cloud is an agglomeration of points in three dimensional space often resembling a cloud, depending on the angle and distance we look at it. Below, you see such a specimen.

{% include figures/happy_buddha.html %}

From the given perspective, there is not a lot to see or understand. If you haven't already, try zooming out now using your mouse wheel (or fingers). As you might notice, a distinctive shape emerges, namely that of the _Happy Buddha_ from [The Stanford 3D Scanning Repository](http://graphics.stanford.edu/data/3Dscanrep/). Here is an image of it from the front:

<div style="text-align:center;">
  <img src="/images/happy_buddha.jpg" style="max-height:700px">
</div>

Here we can already notice the first major differences between the image and point cloud data type: While an image only provides a single _view_ of the object, e.g. from the front, a point cloud can be observed from an arbitrary viewpoint. You can try this by clicking and dragging the point cloud.

Another thing we can notice immediately is the difference in _resolution_ of the image and point cloud. While the image only delivers a single view, it is high quality, showing lots of details. The point cloud on the other hand is more coarse and further doesn't provide any color information[^1]

[^1]: It is possible though to provide each point of a point cloud with color information just like a pixel in an image. This is generally an additional non-trivial step though, and is not directly provided by the sensor, as is the case for most cameras.

The point cloud consist of roughly 30 thousand points, while the image with its $372\times 935$ resolution has a whooping 350 thousand pixels. Interestingly, an image can also be interpreted as a point cloud, but in two dimensional space, as I've shown [here](https://hummat.github.io/learning/2020/07/17/a-sense-of-uncertainty.html#excursus-images). The main difference is, that a pixel in an image is defined by its position on a regular grid, meaning it can be identified by an index, so referring to the pixel at position $(104,517)$ in $x$ and $y$ direction is meaningful. This is not true for a point cloud, where each point can have an arbitrary position and distance to its surrounding points in three dimensional space. This usually leads to point clouds having a larger file size than images, because we need to store three decimal numbers (the $x$, $y$ and $z$ coordinates) for each point, while a pixel only requires two integers[^2]. We usually refer to point clouds as _unordered data_, because if we load the data from a file, it doesn't matter which point we load first or

[^2]: Or even just a single integer---the index of pixel in the array---if we define an order from, e.g., top left to bottom right.

## General stuff

* Problem: How to use deep learning on three dimensional data like point clouds?
* Applications: Computer vision, autonomous driving, robotics, medical treatment
* Tasks: Similar to image data: Classification and segmentation (semantic scene parsing)
* Data comes from laser scanners (LiDAR) or sonar used in robots and autonomous vehicles
* Data representation:
  * Point cloud -> Closest to raw sensor data (so no handcrafted features) but irregularly spaced and unordered
  * Mesh -> Used in computer graphics
  * Volumetric (Voxel) -> Used in 3D CNN, loss of detail due to discretization, ordered like images, large memory usage
  * Projected view (RGB(D)) -> Needs to be taken from various angles which increases the size of the data significantly
* Scale is invariant in 3D contrary to 2D images where scale invariance needs to be learned
* Perspective, illumination, viewpoint effects are diminished in 3D or not present
* Less overlap of bounding boxes in 3D than in 2D
* Fixed size bounding boxes without regression can be used, as objects of each class always have similar size regardless of distance to the sensor

## Learning on point clouds

* PoinNet: End-to-end learning for scattered, unordered point data
  * Challenges:
    * Unordered data: Model needs to be invariant to N! permutations. Different to images where order matters (the position of the pixel in the 2D array is identical to its position in the image).
    * Rotational invariance
  * Solutions:
    * Unordered data -> Symmetric functions, e.g. $f(x_1, x_2,...,x_N)=f(x_2,x_N,...x_1)$. Achieved through $f=\gamma\circ g$ which is symmetric if $g$ is. Max pooling is used together with MLP.
    * Rotational invariance -> Transformer Network: Sub-networks that learns $3\times3$ transformation matrix with regularization term to stay close t orthogonal (so only small transformations are allowed from layer to layer).
  * Results:
    * State-of-the-art
    * Very robust to missing data (50% missing data with only 2% accuracy drop)
    * Achieved through critical points: Only outer hull is used
* PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space
  * Applies PointNet recursively to capture local structure of point clouds. Similar to receptive field in CNNs.
  * Adds a new type of layer to deal with varying densities of point clouds by adaptively combining features from multiple scales.
  * CNNs can perform an overlapping partitioning of the input space through the use of kernels of larger size then the step size. Due to varying densities in point clouds, such a partitioning is achieved through the farthest point sampling algorithm.
  * Input points are randomly dropped out during training to increase robustness to varying density

* Dynamic Graph CNN for Learning on Point Clouds
  * Instead of working on individual points, DGCNN constructs graphs of k-NN around each point and performs convolutions on the edges between the current point and its neighbors (like a kernel in a CNN with one center weight and e.g. 8 surrounding weights).
  * 