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