---
layout: post
title: Making Models from Images
abstract: If you now think, "Photogrammetry? Boring!", or even "SfM & MVS pipeline? Old school!", or maybe even "NeRF? Cool, but cold coffee", behold. Here we will make use of old and new techniques to obtain the best possible result under difficult conditions.
tags: [photogrammetry, sfm, mvs, mvr, deep learning, ycb, bigbird]
category: learning
image: test.png
mathjax: false
jquery: true
plotly: true
time: 0
words: 0
---

# {{ page.title }}

I'll start this article off with a brief motivation for this project. If you are not interested in this, just skip to the [second section](#structure-from-motion-and-multi-view-stereo).

## 1. Motivation

If you have read my [previous article]() you will know that I'm currently trying to improve upon 3D shape completion methods to make them work in the real world, which mostly means on single depth views (i.e depth maps/images). For implicit function models the training data consists of partial point clouds obtained by projecting the depth maps into 3D space using the known camera intrinsic parameters. One further needs a 3D model (i.e. a mesh) to compute occupancy (inside/outside) for arbitrary points in 3D space and its 6D pose, i.e. position and rotation, to align it to the depth data.

Such data is quite expensive to acquire in the real world and thus only limited amounts exist which is not ideal for deep learning methods. Fortunately, using photorealistic synthetic image generation pipelines like [BlenderProc](), one can start from abundant mesh data, like e.g. [ShapeNet](), and render limitless amounts of depth data from them. Unfortunately, to evaluate the performance on real data, which is quite necessary due to the remaining gap between simulated (training) and real (test) data, one still needs at least a decent amount of it.

Enter the [YCB dataset](). It consists of 77 objects from different categories which are placed on a turntable which is rotated in three degree increments and captured with 5 high-res RGB cameras and 5 RGB-D cameras arranged in a quarter circular arc.

Todo: add image of setup

Thus, there are 600 high-res RGB and further 600 RGB-D images per object. Not bad. Additionally, most object were scanned with another high-res scanner to obtain high-quality meshes. Excellent! However, not all objects could be faithfully reconstructed with this setup due to problems of depth sensors to properly perceive semi-transparent and reflective surfaces.

Todo: Show examples

The number of objects is also quite limited when one is only interested in certain categories. Interestingly, there exists a very similar dataset, [BigBird](), with further 125 objects captured with the same setup, except for the high-res mesh generation step. Damn. Here, the problems of low-cost depth scanners used for 3D reconstruction from depth fusion becomes painfully apparent. The question is therefore, can we do better using _photogrammetry_, the act of generating 3D geometry from 2D images, or, as it is known in nerdier circles: the _Structure from Motion_ (SfM), _Multi-View Stereo_ (MVS) pipeline? Let's find out.

## 2. Structure from Motion and Multi-View Stereo

## Code & References

| [Code](/url/to/notebook.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](/url/to/binder/notebook.ipynb) |
|:------------------------------:|:-------------------------------------------------------------------------------:|
|                                |                                                                                 |
|              [1]               |                    [Title of a Reference Paper](/url/to/paper)                  |
