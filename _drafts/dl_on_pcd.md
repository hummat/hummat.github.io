---
layout: post
title: 3D Deep Learning
abstract: A short text explaining what this post is about.
tags: [tag1, tag2, tag3]
category: category
mathjax: false
update: 0000-01-01 
time: 0
words: 0
---

# {{ page.title }}

* Problem: How to use deep learning on three dimensional data like point clouds?
* Tasks: Similar to image data: Classification and segmentation (semantic scene parsing)
* Data comes from laser scanners (LiDAR) or sonar used in robots and autonomous vehicles
* Data representation:
  * Point cloud -> Closest to raw sensor data (so no handcrafted features)
  * Mesh
  * Volumetric (Voxel) -> Used in 3D CNN
  * Projected view (RGB(D))
* PoinNet: End-to-end learning for scattered, unordered point data
  * Challenges:
    * Unordered data: Model needs to be invariant to N! permutations. Different to images where order matters.
    * Rotational invariance
  * Solutions:
    * Unordered data -> Symmetric functions, e.g. $f(x_1, x_2,...,x_N)=f(x_2,x_N,...x_1)$. Achieved through $f=\gamma\circ g$ which is symmetric if $g$ is. Max pooling is used together with MLP.
    * Rotational invariance -> Transformer Network: Sub-networks that learns $3\times3$ transformation matrix with regularization term to stay close t orthogonal (so only small transformations are allowed from layer to layer).
  * Results:
    * State-of-the-art
    * Very robust to missing data (50% missing data with only 2% accuracy drop)
    * Achieved through critical points: Only outer hull is used