---
layout: post
title: Learning from Projections
abstract: The final post in this four-part series on learning from 3D data. How do we learn from 3D data in this approach? We don't. Instead, we project it into the more familiar 2D space and then proceed with business as usual. Neither exciting nor elegant but embarrassingly simple, effective and efficient.
tags: [projection, deep learning, 3D]
category: learning
thumbnail: /images/dragon_projection.png
mathjax: True
time: 0
words: 0
---

# {{ page.title }}
There we are. Welcome to this final part of a four-part series on _learning from 3D data_. After racking our brains to understand deep learning techniques in various 3D representations ([point clouds](https://hummat.github.io/learning/2020/11/03/learning-from-point-clouds.html), [voxel grids](https://hummat.github.io/learning/2020/12/17/learning-from-voxels.html), [graphs and meshes](https://hummat.github.io/learning/2020/12/22/learning-from-graphs.html)), in this last episode we dial it back a notch, more precisely from three to two dimensions.

## Why (not) to project
As always, the first question is, _why should we want to do this?_. Before we can answer this question, let's first define what is meant by _projection_ in this context[^1]. In general, a projection is a _mapping_ transforming something to something else. For geometric settings, i.e. when talking about mappings from one dimension to another, this is often referred to as projecting data from one representation to another. Depending on your background, you might think about dimensionality reduction techniques like PCA, projecting the data from a high to a lower dimensional space, neural networks, projecting (relatively) low dimensional inputs like images into higher dimensional feature space or photography, projecting our three dimensional perception of the world into the two dimensional image plane.

Coincidentally, this last example, photography, is the basis for many projection based learning algorithms in this article. To understand why, simply think about the history of computer vision. For the longest time, this field was confined to two dimensions, as the only commodity sensor capturing visual information was the camera. Only in recent years have 3D scanning devices become more affordable and commonplace due to applications in augmented reality (gaming consoles) and autonomous driving. Besides, there is another reason why it feels natural to use two dimensional data, because its what humans to by default. Yes, we have two eyes, so there is some stereo and thus 3D processing going on, but even if you close one eye, you can still understand your surrounding perfectly well, even though you only work with 2D projections of 3D objects onto your retina.

{% include figures/dragon_3d_2d.html %}
<div style="text-align: center;">
<figure style="width: 45%; display: inline-block;">
    <figcaption>A <b>3D object</b>, converted into a point cloud using a laser scanner.</figcaption>
</figure>
<figure style="width: 45%; display: inline-block">
    <figcaption>The scanned object as a <b>2D projection</b> (an image of the scan)</figcaption>
</figure>
</div>

As a result, there is a huge body of research on how to extract the most information per pixel while at the same time reducing the computational overhead, giving rise to effective and efficient models that can often run in real-time, which is crucial for many real world applications. Thus, it is tempting to make use of this mature approach and translating it to three dimensions. The main advantages are a structured representation which allows to apply our beloved convolutions while also being computationally more efficient, as there is less data to be processed.

So how does this actually work? Below you see the same dragon statue introduced above (taken from [The Stanford 3D Scanning Repository](http://graphics.stanford.edu/data/3Dscanrep/)), both as an actual object in 3D space as well as its 2D projection as seen from the position of the black dot (marked "eye").

{% include figures/projection.html %}

Conceptually, this projection is easy to understand: From every point of the object we want to capture, draw a straight line (green) to the position of the observer. Add it to the image plane (black square), where the plane and the line intersect. This procedure is described mathematically through the _pinhole camera model_, which I won't cover here, but which is implemented in the code for this article to create the figure above, so please [take a look](https://github.com/hummat/hummat.github.io/tree/master/notebooks/learning-from-projections.ipynb) if you are interested.

As a general rule in life, you can't have the cake and eat it too, and projection based 3D deep learning is no exception. In many cases, less data translates into less information. While a three dimensional representation of an object provides a complete picture of all of its features, projecting it to two dimensions, e.g. by taking a picture from one side, can't tell us anything about the opposite side. There are ways to mitigate this shortcoming, but the important word here is _mitigate_. Annoyingly, the more information you try to capture the more dwindles the computational advantage of your approach, in other words, it's a trade off. Let's now see some popular ways to walk this tightrope.

## How to project
The most intuitive and straight forward way to capture more information is to take multiple images from various viewpoints. If you think about, that's exactly what we do when inspecting objects from all sides using our hands (if the object is small) or our legs (if the object is large). Due to self-occlusion (where parts of an object cover other parts of it), we might need a large number of images though, and the computation required to process the information scales linearly with the number of different views.



[^1]: We will take a closer look at different ways to project in the next section.

---