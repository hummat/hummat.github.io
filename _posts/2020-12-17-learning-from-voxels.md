---
layout: post
title: Learning from Voxels
abstract: The second post in the series on learning on 3D data. Last time we looked at point clouds; this time we'll be looking at voxel grids. Let's get to it.
tags: [voxel, 3D, deep learning, voxnet, octnet]
category: learning
image: https://assets.hummat.com/images/3d_conv.png
jquery: true
plotly: true
mathjax: true
time: 9
words: 2250
---

# {{ page.title }}

Welcome to part two of this four part series on learning from 3D data. In the previous post we've seen how to learn from point clouds after some motivation why we would want to and which obstacles need to be overcome in order to do so. Here, we will have a look at another way to represent and work with 3D data, namely the _voxel (**vo**lumetric **el**ement) grid_. The agenda remains unchanged: First we need to get some background on the voxel representation including advantages and disadvantages. Then we will have to understand how the learning actually works (spoiler: 3D convolutions) and how to put it to use. Finally, we see how to overcome some additional problems and maybe peak into some advanced ideas. A lot to do, but don't worry, I'll do my best to not get bogged down into the nitty gritty and as usual, there will be as many visualizations as reasonably justifiable.

## Minecraft

If you haven't come across voxel grids before, simply think _Minecraft_. In a voxel grid, everything is made up of equally sized cubes, the voxels. Below you see the same object---the _Stanford Bunny_ from [The Stanford 3D Scanning Repository](http://graphics.stanford.edu/data/3Dscanrep/)---represented as a point cloud (left) and inside a voxel grid (right)[^1]. More precisely, the second representation is referred to as _occupancy grid_, where only the occupied voxels are displayed. This is easy to obtain from other representations like point clouds by storing a binary variable for each voxel, setting it to $1$ for each voxel which contains at least one point. This corresponds to a black and white image in the 2D domain, i.e. there is a single channel (as opposed to three for the amount of red, green and blue in each pixel) and there are only two _"colors"_, or states, `black` ($0$) and `white` ($1$).
You can drag to rotate and zoom in to reveal individual points and voxels.

<a name="figure"></a>
<div data-include="https://assets.hummat.com/figures/pcd_vs_voxel.html"></div>
<div style="text-align: center">
<figure style="width: 45%; display: inline-block;">
  <figcaption style="text-align: center;  line-height: 1.2em;"><b>Point cloud bunny</b></figcaption>
</figure>
<figure style="width: 45%; display: inline-block">
  <figcaption style="text-align: center; line-height: 1.2em;"><b>Voxel grid bunny</b></figcaption>
</figure>
</div>

While conceptually simple, this binary representation is not the _only_ one. Just as each pixel in an image[^2] can be binary, grayscale (each pixel value lies between $0$ and $1$), RGB colored (three values, each between $0$ and $1$) or even RGB-D, where D corresponds to the depth, i.e. the distance to the sensor (so we are at four values per pixel now), each voxel in a voxel grid can be described by an arbitrary number of values, also called _features_.
For example, instead of setting each voxel to $1$ as soon as a single point happens to be inside, we could instead use some linear interpolation where more points correspond to a value closer to $1$ while a single point corresponds to a value close to $0$.

Another typical extension are _normals_, i.e. vectors orthogonal to the surrounding surface, which can be associated with each voxel by averaging the normal direction of all points residing within. There is an infinite number of things one can try, but the good news is, that it doesn't matter too much[^10]. As soon as we throw a data representation at a deep neural network, it will extract its own features from it and usually it does a much better job than we ever could on our own, which is the whole point of using them in the first place.

Now that we are up to speed on voxel grids, let's move on to the question why we would want to use them as opposed to other kinds of 3D representations and also why we would rather not.

## Convolution! Out of memory…

It all starts with structure. As it turns out, structured information is not only good for computation, where stuff we want to access should be stored in adjacent blocks of memory to speed up retrieval, but it is also good for learning.

### Convoluted information

Going back to images, we find that adjacent pixels are usually highly correlated while far away ones are not. This means knowing about one pixel provides some amount of information about its neighbors[^3]. Now, we can extract this neighborhood information by applying a convolution, i.e. a weight matrix, or kernel, to a patch of the image. As the _"information"_ is represented by pixel values, performing arithmetic on those values, like multiplication and addition, corresponds to information processing, because different pixel and weight values produces different results. For example, keeping the weight matrix constant, as is done for inference, i.e. after the network is trained, we can extract _"edge information"_ from the image by convolving it with an appropriate kernel (e.g. zeros on the left, ones on the right to extract vertical edges).

Crucially, one filter can extract the same information from _everywhere_ in the image, meaning we only need one _"vertical edge detection kernel"_ to extract _all_ vertical edges[^4]. If you wonder why we can't apply the same trick on point clouds have a look at [this section from the previous article](/learning/2020/11/03/learning-from-point-clouds.html#cant-be-that-hard-right). Hint: Point clouds are _unstructured_ due to varying density and permutation invariance. Luckily though, we _can_ apply convolutions on voxel grids, as they are simply three dimensional extensions of 2D pixel grids, i.e. images.

### Some numbers

In the simplest case, you have a grayscale "image", say of size $5\times5$ and a convolutional layer with a single filter, e.g. of size $3\times3$. Each filter has as many kernels as the input has channels, so in this case one. Adding a third input dimension changes almost nothing. Instead of a plane of $5\times5$ pixels we now have a cube with $5\times5\times5$ voxels. Our filter also becomes three dimensional, i.e. $3\times3\times3$. The resulting _feature map_, i.e. the result of convolving the filter with the input, is of size $4\times4$ assuming a stride of one and zero padding in case of the image, and of size $5\times5\times5$ for the voxel grid (where the zero padding is depicted as empty voxels). Easy.

<div data-include="https://assets.hummat.com/figures/3d_conv.html"></div>
<div style="text-align: center">
<figure style="width: 80%; display: inline-block;">
  <figcaption style="text-align: left;  line-height: 1.2em;"><b>3D Convolution:</b> A $5\times5\times5$ dimensional input (greenish) is convolved with a $3\times3\times3$ kernel (gray, red outline) by moving the kernel over all input voxels to produce a $4\times4\times4$ dimensional output (yellowish). Brighter colors encode higher value magnitude (and are only used for visualization purposes, as we are still dealing with a single input and output channel).</figcaption>
</figure>
</div>

Adding more input channels or convolutional filters only slightly complicates things. In 2D, having multiple filters in a convolutional layer results in multiple feature maps, which are usually visualized as a the 3D equivalent of a rectangle (according to Wikipedia, choose one of _rectangular prism_, _rectangular cuboid_ or _rectangular parallelepiped_[^5]). Having multiple 3D filters produces a 4D output, which might sound scary at first, but if you have worked with arrays before, simply think of another box where you toss in all those 3D cubes. For example, having $128$ filters results in an output of size $128\times5\times5\times5$[^6].

### Problem 1: Size

Actually computing this product segues nicely into the downsides of voxel grids, one of which you can easily make out in the figure [above](#figure) and another which is not immediately visible. Let's start with the latter. As mentioned before, a voxel grid encodes occupied _and_ unoccupied space. This is necessary to preserve the position of each voxel in space as the position in the underlying data structure. For example, accessing voxel $[0, 0, 0]$ returns the first voxel in the grid, e.g. the upper left corner in the front, while voxel $[16, 16, 16]$ is found in the lower right corner in the back[^7]. You can try it out by hovering over the voxelized version of the bunny and finding the actual order being used there[^8]. Now, what's not shown are all of the _unoccupied_ voxels. To represent the bunny in an actual voxel _grid_ we would need at least $16\times16\times16=4096$ voxels!

<div data-include="https://assets.hummat.com/figures/voxel_grid.html"></div>

What you see above is an extremely downscaled version of the bunny shown from the front, this time also showing the unoccupied voxels as empty cells. This doesn't look too hard computationally, right? Now try rotating the figure (click and drag).
That's a lot of cells for such a bad representation of our bunny. Now imagine representing a small city scene, needed for applications like autonomous driving, as a voxel grid. We could make the voxel size larger, leading to a coarser grid with less voxels, but this highlights the second major problem of voxel grids: _Discretization_.

### Problem 2: Information

Discretizing continuous space leads to loss of information and aliasing artifacts. This can be seen immediately in the [first example](#figure). While the point cloud version is detailed and smooth, the voxel grid version loses a lot of information. We can of course decrease the voxel size to preserve more information, but we pay for this cubicly in memory consumption and compute.

## Voxelized Deep Learning

In this section, we will see how voxel grids have been used in the literature and how research has tried to make them more well behaved. Let's start with the pioneering approach: _VoxNet_[^9].

### Let's do the obvious: VoxNet

So you have some 3D data lying around and now you want to extract some information from it, say classify objects in it. Assuming you have successfully transformed your 3D data into a voxel representation, now what? On images you'd now what to do: Apply some off-the-shelf deep neural network architecture, maybe pretrained on ImageNet, and you're mostly done. Now extend your 2D to 3D convolutions, stack some of them and throw in some fully connected layers and voila, [_VoxNet_](http://dimatura.net/publications/voxnet_maturana_scherer_iros15.pdf) is born.

<div style="text-align: center">
<figure style="width: 70%; display: inline-block;">
  <img src="https://assets.hummat.com/images/voxnet.png">
  <figcaption style="text-align: center; line-height: 1.2em;"><b>VoxNet</b> [<a href="https://dimatura.net/publications/voxnet_maturana_scherer_iros15.pdf">source</a>]</figcaption>
</figure>
</div>

While not _the_ first to apply 3D convolutions on voxelized 3D data---that honor goes to _Wu et al._ and their [_3D ShapeNets_](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Wu_3D_ShapeNets_A_2015_CVPR_paper.pdf) as far as I know---VoxNet certainly was one of the first to do so, and due to its conceptual simplicity and effectiveness it remains a memorable baseline. Due to computational constraints, point clouds and CAD models were downsampled into $32\times32\times32$ dimensional grids. On large scenes, a sliding window approach needed to be utilized, where small blocks of the scene were processed one at a time.

Convolutions are inherently translation but not rotation invariant, so data augmentation was used during training, presenting rotated versions of each object to the network.

To design an effective and efficient architecture, random search was used, as there wasn’t (and still isn't) a rich pool of provably effective network architectures available in the 3D domain.

Lastly, do simultaneously retain spatial context for large objects while also retaining enough detail for small objects, fused networks outputs on different resolutions were employed.

### Adaptive resolution: OctNet

As discussed in the beginning, and painfully apparent in VoxNet, computational intractability is the Achilles heel of high-res voxel grids, which in turn is the only way to combat loss of information due to discretization. Or is it?

The authors of [_OctNet_](https://openaccess.thecvf.com/content_cvpr_2017/papers/Riegler_OctNet_Learning_Deep_CVPR_2017_paper.pdf) found a, in my eyes, beautiful solution to this problem using _octrees_. An octree is the 3D equivalent of a quadtree in two dimensions. The idea is the same: Partition space into ever smaller areas by adding four (or eight in 3D) new cells to each existing cell, but, and this is the crucial part, _only_ for those cells that actually still contain some information, i.e. are not empty. Here is what that looks like:

<div style="text-align: center">
<figure style="width: 70%; display: inline-block;">
  <img src="https://assets.hummat.com/images/octree.png">
  <figcaption style="text-align: center; line-height: 1.2em;"><b>Octree</b> [<a href="https://openaccess.thecvf.com/content_cvpr_2017/papers/Riegler_OctNet_Learning_Deep_CVPR_2017_paper.pdf">source</a>]</figcaption>
</figure>
</div>

In our case, information can be a point from a point cloud or a face, vertex or edge from a mesh. This allows us to encode empty space by a few large voxels while occupied space is encoded by ever finer voxels, depending on the level of detail present at that particular position. Take a look at the example below.

<div style="text-align: center">
<figure style="width: 70%; display: inline-block;">
  <img src="https://assets.hummat.com/images/octnet.png">
  <figcaption style="text-align: center; line-height: 1.2em;"><b>OctNet</b> [<a href="https://openaccess.thecvf.com/content_cvpr_2017/papers/Riegler_OctNet_Learning_Deep_CVPR_2017_paper.pdf">source</a>]</figcaption>
</figure>
</div>

This concept was used throughout the network, i.e. input _and_ feature space was recursively partitioned, allowing for deep networks and high(er) resolution inputs. This in turn lead to great improvements in accuracy, especially in demanding tasks like object or part segmentation.

## Conclusion

I hope this quick tour provided you with some intuition and understanding of 3D deep learning on structured data and its advantages and downsides. There is of course a lot more to learn and to explore so if your curiosity is not satisfied yet, you could consider taking a look at [more advanced architectures](https://arxiv.org/pdf/1608.04236.pdf) (keyword: _ResNets_) or [training procedures](https://arxiv.org/pdf/1604.03351.pdf) (keyword: _Auxiliary Losses_). Next up: Deep learning on graphs and meshes. Stay tuned!

[Code](https://github.com/hummat/hummat.github.io/blob/master/notebooks/learning-from-voxels.ipynb): [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/hummat/hummat.github.io/HEAD?filepath=%2Fnotebooks%2Flearning-from-voxels.ipynb)

[^1]: The depth is colorcoded to improve interpretability.
[^2]: Have a look [here](/learning/2020/07/17/a-sense-of-uncertainty.html#excursus-images) for an introduction to image representation for computer vision.
[^3]: As discussed [here](/learning/2020/10/16/flatlands.html#images-vs-point-clouds).
[^4]: This is called _weight sharing_ and makes convolutional neural networks so much more efficient than fully connected ones.
[^5]: Definitely the last imho.
[^6]: Using _channels first_ convention.
[^7]: The exact ordering scheme is of little importance, what _is_ important though is that such an order _exists_.
[^8]: Hint: The order is from left to right ($x$), bottom to top ($y$) and front to back ($z$).
[^9]: Name and approach are equally creative.
[^10]: Or at least it shouldn't.

---
