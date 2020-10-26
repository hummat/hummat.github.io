---
layout: post
title: 3D Deep Learning
abstract: In the previous article we've explored 3D data and various ways to represent it. Now, let's look at ways to learn from it to classify objects and perform other standard tasks, common in computer vision, but now in three instead of two dimensions!
tags: [point clouds, voxel grids, meshes, 3D, deep learning]
category: learning
mathjax: true
time: 0
words: 0
---

# {{ page.title }}

It's a great achievement to have found an algorithm that can solve problems on its own (more or less) by simply dropping a huge pile of data on it and telling it what we want to know. In the grand scheme of things this is known as _machine learning_ though nowadays it mostly means _deep learning_.

Deep learning is named after its champion, the _deep neural network_, which enjoys a formidable renaissance since a couple of years. Mostly though, the deep learning revolution has been taking place in a very specific domain, namely that of _images_, which we could also refer to as _2D RGB point clouds_, as you might recall from the [previous post](https://hummat.github.io/learning/2020/10/16/flatlands.html).

From our perspective though, the world is arguably more three dimensional--or even four if you're Einstein--than flat. Naturally, one might wonder why we haven't heard more about advances in 3D deep learning. For once, the data acquisition is not as smooth yet compared to images. Almost everyone has access to a rather high quality camera by reaching into their pockets, but only in recent years have sensors to capture the third dimensions become somewhat affordable in the form of RGBD (D for _depth_) cameras in gaming consoles.

And then there is the computational overhead. Adding a dimension to our data increases computational demands exponentially, rendering most real world tasks infeasible. This is slowly beginning to change though, so it might be an excellent time to dive into this exciting field of research, which I intend to do in this article. We'll be looking a various ways of learning from 3D data which have been proposed over the last couple of years, sorted by the way they represent the third dimension: Point clouds (1), Voxel grids (2), Meshes and Graphs (3) and multiple (2D) views (4). Unsurprisingly, we start with number one, the point cloud.

## 1. Learning from Point Clouds

The pioneering architecture to learn directly on point clouds is [PointNet](https://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf). As discussed in the previous article, point clouds are great, as they are the most natural data representation, directly obtained from our sensors. They are inherently sparse and therefore space-saving, but they also come with some problems. The first is _varying density_, i.e. some regions feature significantly more points than others, and secondly _lack of order_. Both effects are visualized below. Contrary to images, the shape of the point clouds, and therefore the object it represents, is preserved no matter in which order its points are stored and presented. Shuffling the pixels of an image on the other hand usually results in garbage.

{% include /figures/bunny_pcd.html %}

On the left, you see the point cloud of the [Stanford Bunny](http://graphics.stanford.edu/data/3Dscanrep/)[^1]. Points are distributed uniformly over its entire surface. On the right you see the same bunny, but with two important differences: 

1. Points vary in density, leaving some areas empty while others are covered well. This resembles the kind of point clouds obtained from range scans in the wild more closely.
2. The order of the points has been reversed. Imagine we color each point by the position in the $N\times3$ vector in which the point cloud is stored (this means we assume $N$ points with $3$ dimensions, i.e. $x, y, z$ coordinates each) and that point $1$ is closest and point $N$ is furthest away (which produces the _depth_ image, where close points are bright and far points are dark). Reversing the order of the points in the vector but keeping the coloring scheme, we now get dark points in front and bright points at the rear, though the shape of the bunny is unaffected. In fact, any other permutation of the points in the vector would give the same shape (but different coloring), because a point in a point cloud is not defined by its _position in the underlying data structure_ but by its _coordinates in space_. This is what's meant by the term _permutation invariance_ and its a concept our deep learning algorithm needs to learn in order to classify point clouds robustly.

[^1]: I figured as we move to the academic setting in this post, we might as well use the standard academic example instead of the Happy Buddha from the last post.

Due to these difficulties, most previous approaches pre-processed the point clouds either into voxel grids, meshes or collections of 2D images from multiple perspectives (views) to then use more traditional 2D and 3D convolutional neural networks on them. How did the creators of PointNet solve those problems?

To begin with, they only focused on the second problem, permutation invariance, tackling varying density in their second work, which we'll come to later.