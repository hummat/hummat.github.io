---
layout: post
title: Flatlands
abstract: I've recently started working with DLRs robots which perceive their environment not only with cameras but also with depth sensors. Working with the 3D data obtained from these sensors is quite different from working with images and this is the summary of what I've learned so far. How to do deep learning on this data will be covered in the next post.
tags: [point clouds, voxel grids, meshes, 3D]
category: learning
mathjax: true
time: 0
words: 0
---

# {{ page.title }}

While working with the robots at DLR (the German aerospace centre), I've been confronted with a new type of data---next to camera images---which I hadn’t come across so far, namely _point clouds_. As it turns out, point clouds can be an extremely useful extension to the two dimensional RGB camera images already commonly used in scene analysis, for example for object recognition and classification.

However, there are differences between the data types which prevent us from directly applying successful techniques in one area to another. In this post, I'd like to explore those properties after a detailed look at point clouds themselves, to then see which ideas have been employed to extend the deep learning revolution to this promising data type.

## What exactly _is_ a point cloud?

As the name suggests, a point cloud is an agglomeration of points in three dimensional space often resembling a cloud, depending on the angle and distance we look at it. Below, you see such a specimen.

{% include figures/happy_buddha.html %}

From the given perspective, there is not a lot to see or understand. Now, if you haven't already, try zooming out using your mouse wheel (or fingers). As you might notice, a distinctive shape emerges, namely that of the _Happy Buddha_ from [The Stanford 3D Scanning Repository](http://graphics.stanford.edu/data/3Dscanrep/). Here is an image of it from the front:

<div style="text-align:center;">
  <img src="/images/happy_buddha.jpg" style="max-height:700px">
</div>
### Images vs. point clouds

At first glance, images and point clouds seem to be two very different things. Actually though, the underlying concept is the same, they are usually just _represented_ in a different way. It all begins with a bunch of points[^1]. A point can be completely defined by its position in a coordinate frame. For images, this coordinate frame is usually two dimensional, while for point clouds it usually has three dimensions. Below on the left you see a (slightly pixelated version) of the happy Buddha image from before. If you hover over it, you’ll see, for each point, its position in the grid and its RGB color value.[^2] A point defined by a position in a two-dimensional grid, i.e. by it’s position _relative_ to its neighbors rather than by _coordinates_ in 2D space, is called a _pixel_[^3]. For example, a pixel with position $(10, 51)$ is the pixel coming _after_ pixel $9$ and _before_ pixel $11$ in horizontal (or $x$, row) direction and identically so for the vertical (or $y$, column) direction.

[^1]: How those points can be obtained is not part of this post but have a look at _depth cameras_, _time of flight (ToF) cameras_ and _LiDARS_ if you are interested.

{% include figures/image_vs_pcd.html %}

In contrast, the second image (right), shows _the same data_, but now represented as a 2D point cloud. Here, each point is defined by a two-dimensional coordinate, independent from its neighbors. To highlight the difference, I’ve removed _“empty”_ space, i.e. (almost) black pixels, _“converted”_ the grid positions into coordinates (by arbitrarily dividing them by 10) and changed the shape of each point to filled circles instead of squares which are typically chosen to represent pixels. Here, the point at coordinates $(1.0,5.1)$ (the same as the example _pixel_ at $(10,51)$ from before) doesn’t care about its neighbors and doesn’t tell us anything about them. Maybe there is another point at $(1.1, 5.2)$, maybe not, we can’t tell just by knowing about the coordinates of the current point. You can zoom in on both representations (by clicking and dragging a rectangle) to further explore the representational differences.

[^2]: I’ve introduced this kind of image representation [here](https://hummat.github.io/learning/2020/07/17/a-sense-of-uncertainty.html#excursus-images) if you would like to explore this further.
[^3]: Apparently originating from _picture element_.

Of course we might wonder now: if an image can become a point cloud, can a point cloud become something more akin to an image? The answer is, absolutely! Enter _the voxel grid_. A voxel[^4] is for a point in three dimensions, what a pixel is for a point in two dimensions. Just as with images, the position of a voxel in 3D space is defined by its position in the underlying data structure, a 3D grid, relative to its neighbors. Below you see the voxel representation of the happy Buddha point cloud.

[^4]: _volume_ or _volumetric_ _element_

{% include figures/pcd_as_voxel.html %}

To create it, we simply divide the space into cubes of equal size (the voxel grid) and only display those cubes (voxels) which consume at least a single point from our point cloud.

The final 3D data representation I want to introduce in this post is the _mesh_. 

Each representation has its advantages and disadvantages. Point clouds are closest to the raw data we receive from our depth sensors, so no postprocessing (often including hyperparameters like voxel size) is required. They occupy little memory because of their efficient representation as points in 3D space and their natural _sparsity_, where empty space is signified by the absence of points in contrast with voxel grids (as well as images for that matter), where emptiness needs to be explicitly encoded to preserve the grid structure.

Downsides of this representation include the irregularity of which the points occupy the space, i.e. distances between points are generally not identical. Further, point clouds are _unordered_, in contrast to images or voxel grids, where knowing about one voxel (or pixel) provides information about its neighbors.

For voxel grids we have already discussed that they are more structured than point clouds (which means we can use the extension of 2D convolutions when training neural networks on them) but are significantly less memory efficient. Another obvious downside is the loss of information when discretizing the space into cubes (compare the point cloud representation of our happy Buddha to its voxel representation to see this effect).

What about meshes then?

{% include figures/pcd_as_mesh.html %}

## General stuff

* Problem: How to use deep learning on three dimensional data like point clouds?
* Applications: Computer vision, autonomous driving, robotics, medical treatment
* Tasks: Similar to image data: Classification and segmentation (semantic scene parsing)
* Data comes from laser scanners (LiDAR) or sonar used in robots and autonomous vehicles
* Data representation:
  * Point cloud -> Closest to raw sensor data (so no handcrafted features) but irregularly spaced and unordered
  * Mesh -> Used in computer graphics
  * Volumetric (Voxel) -> Used in 3D CNN, loss of detail due to discretization, ordered like images (no space between pixels/voxels), large memory usage
  * Projected view (RGB(D)) -> Needs to be taken from various angles which increases the size of the data significantly
* Scale is invariant in 3D contrary to 2D images where scale invariance needs to be learned
* Perspective, illumination, viewpoint effects are diminished in 3D or not present
* Less overlap of bounding boxes in 3D than in 2D
* Fixed size bounding boxes without regression can be used, as objects of each class always have similar size regardless of distance to the sensor
