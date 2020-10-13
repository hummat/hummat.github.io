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

From the given perspective, there is not a lot to see or understand. Now, if you haven't already, try zooming out using your mouse wheel (or fingers). As you might notice, a distinctive shape emerges, namely that of the _Happy Buddha_ from [The Stanford 3D Scanning Repository](http://graphics.stanford.edu/data/3Dscanrep/). Here is an image of it from the front:

<div style="text-align:center;">
  <img src="/images/happy_buddha.jpg" style="max-height:700px">
</div>
### Images vs. point clouds

At first glance, images and point clouds seem to be two very different things. Actually though, the underlying concept is the same, they are usually just _represented_ in a different way. It all begins with a bunch of points. A point can be completely defined by its position in a coordinate frame. For images, this coordinate frame is usually two dimensional, while for point clouds it usually has three dimensions. Below on the left you see a (slightly pixelated version) of the happy Buddha image from before. If you hover over it, you’ll see, for each point, its position in the grid and its RGB color value.[^1] A point defined by a position in a two-dimensional grid, i.e. by it’s position _relative_ to its neighbors rather than by _coordinates_ in 2D space, is called a _pixel_[^2]. For example, a pixel with position $(10, 51)$ is the pixel coming _after_ pixel $9$ and _before_ pixel $11$ in horizontal (or $x$) direction and identically so for the vertical (or $y$) direction.

{% include figures/image_vs_pcd.html %}

In contrast, the second image (right), shows _the same data_, but now represented as a 2D point cloud. Here, each point is defined by a two-dimensional coordinate, independent from its neighbors. To highlight the difference, I’ve removed _“empty”_ space, i.e. (almost) black pixels, _“converted”_ the grid positions into coordinates (by arbitrarily dividing them by 10) and changed the shape to points instead of squares which are typically chosen to represent pixels. Here, the point at coordinates $(1.0,5.1)$ (the same as the example _pixel_ from before) doesn’t care about its neighbors and doesn’t tell us anything about them. Maybe there is another point at $(1.1, 5.2)$, maybe not, we can’t tell just by knowing about the coordinates of the current point. You can zoom in on both representations (by clicking and dragging a rectangle) to further explore the representational differences.

[^1]: I’ve introduced this kind of image representation [here](https://hummat.github.io/learning/2020/07/17/a-sense-of-uncertainty.html#excursus-images) if you would like to explore this further.
[^2]: Apparently originating from _picture element_.

Of course we might wonder now: if an image can become a point cloud, can a point cloud become something more akin to an image? The answer is, absolutely! Enter _the voxel grid_. A voxel[^3] is for a point in three dimensions, what a pixel is for a point in two dimensions. Just as with images, the position of a voxel in 3D space is defined by its position in the underlying data structure, a 3D grid, relative to its neighbors.

[^3]: _volume_ or _volumetric_ _element_

Both representations have their advantages and disadvantages. While grid data ()

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