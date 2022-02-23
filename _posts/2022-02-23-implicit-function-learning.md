---
layout: post
title: Implicit Function Learning
abstract: "There are many ways to represent 3D shapes, from the ubiquitous triangle mesh to voxel grids and point clouds. And then there is the one to rule them all: The implicit function."
tags: [3D, deep learning, reconstruction, completion]
category: learning
thumbnail: /images/implicit/suzanne.png
mathjax: true
jquery: true
plotly: true
time: 8
words: 2109
---

# {{ page.title }}

As the title implies, this article touches on two topics:
1. What are implicit functions?
2. How can they be learned from data for arbitrary shapes?

Naturally, we start with the first.

## What is an implicit function?

You are probably familiar with the concept of functions from mathematics and programming. Both have in common that you provide one or several input(s) $x$ and obtain one or several output(s) $y$ as in $f(x)=y$. Because we want to understand functions in the context of 3D shape representation, consider the function describing the surface of a sphere centered at the origin

$$
\begin{align}
    x&=\cos(\alpha)\sin(\beta)r \\
    y&=\sin(\alpha)\sin(\beta)r \\
    z&=\cos(\beta)r
\end{align}
$$

where $\alpha$ and $\beta$ are the azimuth and polar angle and $r$ is the radius of the sphere. For a sphere with radius one ($r=1$) and angles $\alpha=30^{\circ},\beta=55^{\circ}$ we obtain $(x,y,z)=(0.71,0.41,0.57)$.

<div data-include="/figures/implicit/sphere.html"></div>

This is an _explicit_ or _parametric_ way of defining the function: Given the sphere radius, we can directly compute all point coordinates on its surface for arbitrary spherical coordinates $\alpha,\beta$ as $f(\alpha,\beta,r)=(x,y,z)$. Here is the same function defined _implicitly_:

$$f(x,y,z)=x^2+y^2+z^2=r^2$$

This equation provides the squared distance to the spheres surface from any point $(x,y,z)$ which is easier to see if we rewrite it as $x^2+y^2+z^2-r^2=0$ which is the standard form of implicit functions. All points for which the equation returns zero lie on the surface of the sphere. Further, we know that all points for which $f(x,y,z)<0$ must lie _inside_ while those for which $f(x,y,z)>0$ must lie on the _outside_ of the sphere. See the visualization above by clicking on _implicit_. Therefore, our function returns a _signed distance_ and computing it for all points inside the volume we obtain a _signed distance field_ or _SDF_. In doing so, we have implicitly defined the spheres surface as the so called _"zero-level set"_, i.e. the _ISO surface_ at distance zero, a concept you might be familiar with in the context of elevation or ISO _lines_[^1] on topological maps.

## Why are they useful?

Remember the simple but powerful equation describing _all_ points on the surface of a sphere with arbitrary radius introduced in the beginning. Now try to find a similar equation for the following shape[^2]:

<div data-include="/figures/implicit/suzanne.html"></div>

Yeah, me neither. What we still can do rather efficiently though is to compute, for every point, whether it lies within or outside the shape. To do so, we select a second point far off in the distance that's surely not within the shape, and connect it with our point of interest[^3]. We then perform a triangle intersection test (as we do have a triangle mesh of our shape). For every triangle, we first extend it indefinitely to form a plane and compute the intersection of our line (or ray) with it (provided line and plane aren't perpendicular which we can check by computing the dot product of the planes normal vector and the line). If plane and line intersect we check if the point of intersection lies to the left of all triangle edges. If so, the triangle and the line intersect[^4]. Now, an even number of intersections means the point of interest lies outside the shape (we have entered and exited), while an uneven number implies we are on the inside.

Repeating this process many times provides a separation of the space, e.g. a $1\times1\times1$ cube, into exterior (**<span style="color: #EF553B">red</span>**), interior (**<span style="color: #636EFA">blue</span>**) and _in between_ (**<span style="color: lightgray">white</span>**), where _in between_ describes the surface of the shape _implicitly_ as space that's neither inside nor outside the shape.

<div data-include="/figures/implicit/classification.html"></div>

Instead of only computing whether the point of interest is inside our outside the shape we can also compute its distance where a positive distance means outside, and a negative inside as in the sphere example from before[^5]. This is a little more involved to compute for general shapes and often not necessary as we will see later. Below you can see a cross-section the surface of the zero-level set as well as the surrounding levels.

<div data-include="/figures/implicit/regression.html"></div>

## How can they be learned?

Before we dive into this question, let's first quickly cover another: why? If you have dealt with 3D data (in the context of deep learning or otherwise) before, you may know that there are many ways to represent it, unlike for images, where the absolute standard is the pixel grid.

<div style="text-align: center;">
    <figure style="width: 24%; display: inline-block;">
        <div data-include="/figures/implicit/image.html"></div>
        <figcaption><b>Image</b></figcaption>
    </figure>
    <figure style="width: 24%; display: inline-block;">
        <div data-include="/figures/implicit/pcd.html"></div>
        <figcaption><b>Point Cloud</b></figcaption>
    </figure>
    <figure style="width: 24%; display: inline-block;">
        <div data-include="/figures/implicit/voxel.html"></div>
        <figcaption><b>Voxel Grid</b></figcaption>
    </figure>
    <figure style="width: 24%; display: inline-block;">
        <div data-include="/figures/implicit/mesh.html"></div>
        <figcaption><b>Mesh</b></figcaption>
    </figure>
</div>

Each representation has its advantages and shortcomings. While pointclouds are lightweight, they are hard to process due to their disarray. Voxel grids on the other hand are easy to process with standard operations like convolutions but there is a tradeoff between loss of detail and large size. Meshes finally are an efficient representation with great properties for downstream tasks but extremely hard to learn as one needs to keep track of vertices and triangles and their interdependence[^6].

What we would like is a representation that is expressive, low cost (computational as well as storage wise) and easy to learn. Well, you probably see where this is going, but how is it that implicit functions combine all those magnificent properties?

Let's begin from the end: They are easy to learn because they define a simple learning task. Learning whether a point lies on the in- or outside can be readily cast into a binary classification problem (see the _probabilities_ tab in the figure above). The model, in our case a deep neural network, learns to assign a score (aka a logit) to each point inside our predefined volume (the $1\times1\times1$ cube). When passed through a sigmoid function and interpreted as a probability, the surface of the objects is _implicitly_ defined by the _decision boundary_ between the two classes, i.e. at $0.5$ probability.

On the other hand, learning to predict a signed distance to the surface is a classic regression setup. We can simply learn to minimize the difference between the actual and predicted scalar distance value for each point.

Now to understand the other two desirable attributes, let's have a look at a network architecture designed to learn binary occupancy values: The [_Occupancy Network_](https://avg.is.mpg.de/publications/occupancy-networks).

### Occupancy Networks

<figure style="text-align: center;">
    <img src="/images/implicit/occnet.png" alt="Occupancy Network">
    <figcaption><a href="https://avg.is.mpg.de/publications/occupancy-networks"><b>The Occupancy Network</b></a></figcaption>
</figure>

From left to right: We decide on an input data representation. Depending on it, we use an encoder build up from standard operations like 2D or 3D convolutions, fully connected layers, residual connections etc. For example, we could use a ResNet backbone (the convolutional part) pre-trained on ImageNet when dealing with images or a PointNet when the input consists of point sets (pointclouds), i.e. projected depth maps or LiDAR scans.

The output is a _global_ feature vector describing the input. Now here comes the interesting part. We use a decoder consisting of multiple fully-connected layers (a _multi-layer perceptron_ or MLP), hand over the global feature vector and ask it to predict for arbitrary (randomly sampled), continuous point positions to predict their binary occupancy probabilities. Close to $1$ for inside the shape defined by the input and close to $0$ for outside. Doing so many times for all shapes in our training dataset, the MLP learns, or rather, _becomes_ a partition of space based on the input, i.e. a three-dimensional probability distribution of empty and occupied space.

This is the key to both the expressiveness and efficiency of the implicit representation as a (small) MLP can encode hundreds of shape surfaces in a continuous fashion, i.e. arbitrary resolution.

There is one problem though. As the feature vector describes the input globally, there is no spacial correspondence between input, feature and output space, so the produced shapes lack detail and are overly smooth. Luckily, this can be fixed.

### Convolutional Occupancy Networks

<figure style="text-align: center;">
    <img src="/images/implicit/conv_occnet_vol.png" alt="Convolutional Occupancy Network">
    <figcaption><a href="https://is.mpg.de/publications/peng2020eccv"><b>The Convolutional Occupancy Network</b></a></figcaption>
</figure>

Through the division of the input space into voxels, a discrete feature grid can be constructed allowing to correlate input and output spacially with the feature space. As the grid resolution is finite, continuous point locations are matched with inpainted features from a 3D UNet and trilinearly interpolated. The decoder stays the same but is fed with higher quality features resulting in detailed shapes with higher frequency information.

### Implicit Feature Networks

<figure style="text-align: center;">
    <img src="/images/implicit/ifnet.png" alt="Implicit Feature Network">
    <figcaption><a href="https://virtualhumans.mpi-inf.mpg.de/ifnets"><b>The Implicit Feature Network</b></a></figcaption>
</figure>

Finally, a similar idea has been implemented in _Implicit Feature Networks_, but instead of feature inpainting and interpolation on the decoder side, multi-scale features are extracted by the encoder using a 3D feature pyramid, akin to its 2D pendant known from detection and segmentation networks.

### Extracting the mesh

Now, you might be wondering how to actually obtain a mesh from the implicit representation. While during training we asked the model to predict occupancies for random point positions, during inference we instead extract occupancies on a regular grid, at arbitrary resolution. Having obtained this grid where each voxel is either occupied or unoccupied (or filled with a occupancy probability or signed distance value), we can use the classical _Marching Cubes_ method to extract the mesh.

## Are there any problems?

Yes. While most works focus on benchmark data like ShapeNet, the real world is more messy. A promising application of shape completion is in robotic manipulation but the data obtained from the RGB-D cameras is less than ideal. The data is noisy, has missing parts due to overly reflective materials, doesn't come in a canonical pose--both due to the position of the robot relative to the object and the objects pose in the world--and of course comes in very diverse shapes and sizes.

<div data-include="/figures/implicit/problems.html"></div>

## Let's see some results!

Below there are a few visualizations of input pointclouds obtained by projecting depth images and their completions in the form of extracted meshes which you can blend in by clicking on _Mesh_ in the legend of each figure.

<div style="display: flex; justify-content: space-between;">
    <figure style="width: 49%;">
        <div data-include="/figures/implicit/val_result2.html"></div>
    </figure>
    <figure style="width: 49%;">
        <div data-include="/figures/implicit/val_result3.html"></div>
    </figure>
</div>
<div style="display: flex; justify-content: space-between;">
    <figure style="width: 49%;">
        <div data-include="/figures/implicit/val_result5.html"></div>
    </figure>
    <figure style="width: 49%;">
        <div data-include="/figures/implicit/val_result6.html"></div>
    </figure>
</div>

As can be seen, the network is able to complete a quite diverse set of shapes, all coming from the same class (bottles) though. That's all for today. As usual, the code for generating the visualizations can be found below. Be aware though that the data for generating the figures is too large this time, so I won't be including it in the repository.

[^1]: A curve along which a continuous field has a constant value.
[^2]: Introducing _Suzanne_, the mascot of the awesome open source 3D software [Blender](https://www.blender.org).
[^3]: This is sometimes called "shooting a ray from an arbitrary viewpoint". 
[^4]: Check out [this site](https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution) if you are curious and want more details.
[^5]: By hovering over a point you can see its distance to the surface.
[^6]: You can read more on each representation in my previous posts on learning from [point clouds](https://hummat.github.io/learning/2020/11/03/learning-from-point-clouds.html), [voxel grids](https://hummat.github.io/learning/2020/12/17/learning-from-voxels.html), [graphs and meshes](https://hummat.github.io/learning/2020/12/22/learning-from-graphs.html) or [projections](https://hummat.github.io/learning/2021/02/04/learning-from-projections.html).

## Code & References

| [Code](https://github.com/hummat/hummat.github.io/blob/master/notebooks/implicit-function-learning.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/hummat/hummat.github.io/HEAD?filepath=%2Fnotebooks%2Fimplicit-function-learning.ipynb) |
| :----------------------------: | :-----------------------------------------------------------------------------: |
|              [1]               |                   [Mescheder et al.: Occupancy Networks: Learning 3D Reconstruction in Function Space](https://avg.is.mpg.de/publications/occupancy-networks)                   |
| [2] | [Peng et al.: Convolutional Occupancy Networks](https://is.mpg.de/publications/peng2020eccv) |
| [3] | [Chibane et al.: Implicit Functions in Feature Space for 3D Shape Reconstruction and Completion](https://virtualhumans.mpi-inf.mpg.de/ifnets) |
