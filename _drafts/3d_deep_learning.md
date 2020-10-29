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

## 1. Learning on Point Clouds

As discussed in the previous article, point clouds are great, as they are the most natural data representation, directly obtained from our sensors. They are inherently sparse and therefore space-saving, but they also come with some problems. The first is _varying density_, i.e. some regions feature significantly more points than others, and secondly _lack of order_. Both effects are visualized below. Contrary to images, the shape of the point clouds, and therefore the object it represents, is preserved no matter in which order its points are stored and presented. Where we to reverse the order of pixels in an image we would flip it horizontally (i.e. perform a $180°$ clockwise rotation), while shuffling the pixels would results in garbage (noise).

{% include /figures/bunny_pcd.html %}

On the left, you see the point cloud of the [Stanford Bunny](http://graphics.stanford.edu/data/3Dscanrep/)[^1]. Points are distributed uniformly over its entire surface. On the right you see the same bunny, but with two important differences: 

1. Points vary in density, leaving some areas empty while others are covered well. This resembles the kind of point clouds obtained from range scans in the wild more closely.
2. The order of the points has been reversed. Imagine we color each point by the position in the $N\times3$ vector in which the point cloud is stored (this means we assume $N$ points with $3$ dimensions, i.e. $x, y, z$ coordinates each) and that point $1$ is closest and point $N$ is furthest away (which produces the _depth_ image, where close points are bright and far points are dark). Reversing the order of the points in the vector but keeping the coloring scheme, we now get dark points in front and bright points at the rear, though the shape of the bunny is unaffected. In fact, any other permutation of the points in the vector would give the same shape (but different coloring), because a point in a point cloud is not defined by its _position in the underlying data structure_ but by its _coordinates in space_. This is what's meant by the term _permutation invariance_ and its a concept our deep learning algorithm needs to learn in order to classify point clouds robustly.

[^1]: I figured as we move to the academic setting in this post, we might as well use the standard academic example instead of the Happy Buddha from the last post.

A third problem pose rigid motions. Rigid motions are a subset of affine transformations, including translation and rotation. Rotation invariance on images is often improved by augmenting the training data with small random rotations, while translation invariance is an inherent feature of convolutions. Both are infeasible in three dimensions because we're dealing with three rotation axes as opposed to one (requiring exponentially more data augmentation to cover the same range of motion) and we can't convolve the point cloud because it's unordered (see problem 2).

Due to these difficulties, most previous approaches pre-processed the point clouds either into voxel grids, meshes or collections of 2D images from multiple perspectives (views) to then use more traditional 2D and 3D convolutional neural networks on them. Let's now have a look at how those problems have been tackled in research.

### 1.1 PointNet & PointNet++

In the pioneering architecture for learn directly on point clouds, [PointNet](https://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf), the authors only focused on the second and third problem, namely permutation and transformation invariance, tackling varying density in their second work, which we'll come to later.

#### 1.1.1 Solving transformation invariance

If we can't learn something implicitly from data, we need to build it into our model explicitly. In PointNet, this is achieved through a small sub-network which learns to predict a $3\times3$ rotation matrix which is then applied to the point cloud. This allows the network to align each object in a way it sees fit. While the first transformation is applied to the input (in three dimensional input space), it is later repeated in $64$ dimensional feature space (giving rise to a $64\times64$ matrix). Because learning a useful transformation matrix in this high dimensional space is more challenging, the authors constrain it to be close to an orthogonal matrix by adding an $L_2$-like regularization term. These small sub-networks are trained end-to-end with the rest of the network, even though they solve a regression task (finding $9$ and $4096$ real numbers respectively) instead of the overarching classification (or segmentation) task.

Interestingly, this increased the accuracy by "only" $2\%$, but I'm guessing that the training data was already relatively homogeneous and that improvements could be bigger on less uniform or more corrupted real world data.

#### 1.1.2 Solving computational complexity

Now that our point cloud is oriented correctly, we need to classify it. If we were to naively implement a neural network to do so, we would need $N\times3$ input units (neurons), as we can't use convolutions on unstructured data, because they inherently assume that neighboring points (or pixel) are correlated, while, as discussed above, the order of points in a point cloud is arbitrary. Apart from the immense computational complexity for large $N$ (large point clouds with many points), we would have to train one network per point cloud size or sub/super sample each input to have exactly the same number of points (this is commonly done when pre-processing images for classification, though not for semantic segmentation). Below (figure 1) I've visualized how a single neuron (a) and an entire fully connected layer (b) operates on an $28\times28=784$ pixel 2D monochrome (black and white) image, which we will use as a foundation to build up our understanding of the _PointNet approach_. The mathematical operation performed by _each unit_ in this layer is

$$y=\sum_{i=1}^{784}w_ix_i+b,$$

where $b$ is an offset, usually called the _bias_ (not shown in the figure). This means we obtain $784$ scalar outputs $y$ from the layer, one from each unit.

<div style="text-align: center">
<figure style="width: 45%; display: inline-block;">
  <img src="/images/fc_vs_conv/fc_single.png">
  <figcaption style="text-align: left;  line-height: 1.2em;"><b>Fig. 1 (a):</b> A single fully connected neuron, only showing connections (weights $w$) for two inputs (pixel).</figcaption>
</figure>
<figure style="width: 45%; display: inline-block">
  <img src="/images/fc_vs_conv/fc_full.png">
  <figcaption style="text-align: left; line-height: 1.2em;"><b>Fig. 1 (b):</b> A fully connected layer, only showing four of the $784$ neurons and two of the $784$ inputs each.</figcaption>
</figure>
</div>

Another idea would be, to _"slide"_ a network with $3$ inputs, one for each spatial dimensions, over each of the $N$ points[^2]. This is what's meant in the paper where the authors introduce the concept of a _shared MLP_. MLP being _multi-layer perceptron_, i.e. a network with fully connected layers[^3] only. This means we share one network for all $N$ points in the point cloud. This is what the input to a network classifying individual points instead of point clouds would look like. This is visualized below.

<a name="pointnet_mlp"></a>
<div style="text-align: center">
<figure style="width: 90%; display: inline-block;">
  <img src="/images/fc_vs_conv/pointnet_mlp.png">
  <figcaption style="text-align: left;  line-height: 1.2em;"><b>Fig. 2:</b> A "shared" MLP with three inputs ($x,y,z$ coordinates) "sliding" over each point in the point cloud.</figcaption>
</figure>
</div>

Let's have a closer look at _sharing_ and _sliding_. If the notion of sliding weights over inputs, performing multiplications and additions sounds familiar to you, that's because it is the definition of a convolution! But wait, didn't I just discredit convolutions for the use on point clouds? Well, bear with me for a second.

[^2]: This _"sliding"_ notion was initially introduced in the [Network in Network paper](https://arxiv.org/pdf/1312.4400.pdf%20http://arxiv.org/abs/1312.4400.pdf).
[^3]: Also often called _dense_ layers.

In what follows, I'm assuming some basic familiarity with (convolutional) neural networks on your part. If that's not the case, have a look at [the background section](https://hummat.github.io/learning/2020/07/17/a-sense-of-uncertainty.html#some-background) of one of my previous posts or [this excellent explanation](https://cs231n.github.io/convolutional-networks/) from the famous CS231n Stanford course to brush up your knowledge.

To learn from images, we present them, one by one or in batches of multiple images, to a stack of convolutional layers, each consisting of a stack of filters in turn. This exploits the structure of images, where neighboring pixels are assumed to be highly correlated, to reduce the number of parameters compared to a fully connected approach, where each pixel gets its own weight.

A standard convolutional layer is defined by the number of input channels, e.g. the red, green and blue color channels of an image for the input layer, the height and width of the kernels or weight matrices (we have one kernel per input channel) which are convolved with the input channels to give the convolutional neural network its name, and the number of output channels, or feature maps, or filters. A convolutional layer operating on an RGB image with $3\times3$ kernels and 128 filters would therefore be of dimension `input channel`x`kernel height`x`kernel width`x`output channel` i.e. $3\times3\times3\times128$. The first dimension is usually omitted, as it is deemed obvious (i.e. easily inferred from the number of channels from the input) and the output channel are sometimes stated in the first (_"channels first"_) or last (_"channels last"_) position. Below (figure 3) you see a simple convolution on a monochrome (black and white) input image (a) and the conceptually easy to imagine implementation using a "sliding fully connected" network (b).

<div style="text-align: center">
<figure style="width: 45%; display: inline-block;">
  <img src="/images/fc_vs_conv/conv_slide.png">
  <figcaption style="text-align: left;  line-height: 1.2em;"><b>Fig. 3 (a):</b> A standard convolution of a single filter with one $3\times3$ kernel.<br><br></figcaption>
</figure>
<figure style="width: 45%; display: inline-block">
  <img src="/images/fc_vs_conv/fc_slide.png">
  <figcaption style="text-align: left; line-height: 1.2em;"><b>Fig. 3 (b):</b> Conceptually simple: Sliding a "fully connected" layer over the image, restricting it to $9$ adjacent inputs.</figcaption>
</figure>
</div>

If you were to actually implement this though, you would notice that it's a non-trivial task, because both "sliding" and "only being partially connected" are not part of the standard repartoire of a fully connected layer. Instead, let's try to express a fully connected layer as a convolution, which slides and partially connects natively. To do so, we have two options:

1. Using one filter per input pixel with one large kernel ($28\times28$) per input channel (figure 4).
2. Using one filter per input pixel with one small kernel ($1\times1$) per input channel _and_ pixel (figure 6).

<div style="text-align: center">
<figure style="width: 90%; display: inline-block;">
  <img src="/images/fc_vs_conv/conv_full2.png">
  <figcaption style="text-align: left;  line-height: 1.2em;"><b>Fig. 4:</b> The operation of a fully connected layer on a two-dimensional input (only looking at spatial dimensions here, not depth, i.e. number of channels) can be described by a convolution with one kernel (weight matrix) per input channel (one) of the same size as the input ($28\times28$ pixel), repeated for each input pixel (i.e. $784$ filter/output channel).</figcaption>
</figure>
</div>

I think the first approach is relatively intuitive. The convolution operation performed by each filter is identical to that stated above for a single fully connected unit, i.e. multiplying a unique scalar weight with each input pixel and summing them up. Consequently, we obtain the same $784$ scalar outputs, one from each filter, perform the same number of operations (multiplications and additions) and have the same number of parameters ($784\times784=614656$ omitting biases).

If we reduce the width and height of our filter kernels to one, we obtain $1\times1$ convolutions. Let's first look at a trivial example (figure 5) with a single filter and three kernels, operating on an RGB image (a) and again the conceptually simple extension to the fully connected approach (b).

<div style="text-align: center">
<figure style="width: 45%; display: inline-block;">
  <img src="/images/fc_vs_conv/conv_rgb.png">
  <figcaption style="text-align: left;  line-height: 1.2em;"><b>Fig. 5 (a):</b> A standard convolution of a single filter with three $1\times1$ kernels.<br><br></figcaption>
</figure>
<figure style="width: 45%; display: inline-block">
  <img src="/images/fc_vs_conv/fc_rgb.png">
  <figcaption style="text-align: left; line-height: 1.2em;"><b>Fig. 5 (b):</b> Conceptually simple: Sliding a "fully connected" layer over the image, restricting it to one pixel per channel.</figcaption>
</figure>
</div>
The important thing to note here is, that both approaches produce a _single_ output per spatial dimension (i.e. width and height), because, per convention[^4], convolutions of input channels and kernels from the same filter are summed up, meaning each filter only produces one feature map. In the example above, the red, green and blue pixel values are multiplied by the red green and blue weights and then summed to give a scalar output. This is actually the more common way to employ $1\times1$ convolutions, namely as a way to reduce or increase depth, i.e. the number of output channels and not to mimic fully connected operations. For example, we can put this to use in the final layer of our network to produce one feature map per class by sliding $10$ filter with $1\times1$ kernels over the output feature maps of the previous layer. In the figure below (6), this would produce one feature map for each digit from one to ten, where each "pixel" in the feature map corresponds to the "oneishness" or "twoishness" of each input pixel[^5].

[^4]: This part is often omitted!
[^5]: Interestingly, summing those feature maps over the spatial dimensions would again produce the same result as a fully connected layer, i.e. one scalar value per class, so we could view it as a third way of mimicking fully connected layers with convolutions.

<div style="text-align: center">
<figure style="width: 45%; display: inline-block;">
  <img src="/images/fc_vs_conv/conv_slide2.png" style="padding: 50px 0 50px 0">
  <figcaption style="text-align: left;  line-height: 1.2em;"><b>Fig. 6 (a):</b> Using $1\times1$ convolutions to produce "class" feature maps with <em>one filter</em> per class.<br><br></figcaption>
</figure>
<figure style="width: 45%; display: inline-block">
  <img src="/images/fc_vs_conv/fc_slide2.png">
  <figcaption style="text-align: left; line-height: 1.2em;"><b>Fig. 6 (b):</b> The same effect achieved with a fully connected approach, were we to focus on one pixel at a time and capable of sliding the layer across the input.</figcaption>
</figure>
</div>

Armed with the knowledge from the previous paragraph, we can now understand the second approach of transforming a fully connected layer into a convolutional layer. To do so, we first _transform the image into a vector_[^6] by concatenating all of its pixels and then apply _one filter per pixel_ with _one kernel per pixel_, as our image now has as many channels as it had pixels (i.e. it is of shape $1\times1\times784$). Have a look at figure 7 below to take it in visually.

[^6]: Another important information often omitted (or not considered?) in the explanations I found on the subject.

<div style="text-align: center">
<figure style="width: 45%; display: inline-block;">
  <img src="/images/fc_vs_conv/conv_single1.png" style="padding: 50px 0 50px 0">
  <figcaption style="text-align: left;  line-height: 1.2em;"><b>Fig. 7 (a):</b> A $28\times28$ pixel image transformed into a $1\times1\times784$ vector (left) is convolved with a single filter with $784$ one-by-one kernels (right) producing a single scalar value.</figcaption>
</figure>
<figure style="width: 45%; display: inline-block">
  <img src="/images/fc_vs_conv/conv_full1.png">
  <figcaption style="text-align: left; line-height: 1.2em;"><b>Fig. 7 (b):</b> The same input image as in (a), but now convolved with $784$ filters, each with $784$ one-by-one kernels, producing $784$ scalar outputs.<br><br></figcaption>
</figure>
</div>

Because each filter produces a single scalar (as input dimensions are summed), we again end up with $784$ outputs. So it's actually not as simple as exchanging the fully connected layer by a $1\times1$ convolution to gain the same functionality. Instead, we need to first reshape the image and then apply as many filter as there are pixel in the input. Finally, how can this be applied to our point cloud problem?

On the input layer, we can replace our fully connected layer from [figure 2](#pointnet_mlp) with a $1\times3$ convolutions, i.e. 64 filter with one $1\times3$ kernel each, as shown below.

<div style="text-align: center">
<figure style="width: 80%; display: inline-block;">
  <img src="/images/fc_vs_conv/pointnet_conv2.png">
  <figcaption style="text-align: left;  line-height: 1.2em;"><b>Fig. 8:</b> The "shared" MLP from before is replace by $1\times3$ convolutions.</figcaption>
</figure>
</div>

We could also use the second approach employing $1\times1$ convolutions, which actually is used deeper in the network, but this would require us to reshape the point cloud beforehand. To the contrary, once we have employed the technique once (on the inputs), the output is one scalar value per filter, i.e. of dimension $1\times1\times64$, exactly what we need for the second approach!

<div style="text-align: center">
<figure style="width: 60%; display: inline-block;">
  <img src="/images/fc_vs_conv/pointnet_conv1.png">
  <figcaption style="text-align: left;  line-height: 1.2em;"><b>Fig. 8:</b> The "shared" MLP is mimicked by $1\times1$ convolutions inside the network (shown on the input here though).</figcaption>
</figure>
</div>

cloud case (though [the authors implemented it with a single $1\times3$ kernel](https://github.com/charlesq34/pointnet/blob/master/models/pointnet_cls.py) instead of three $1\times1$ kernels). What's called _mlp(64,64)_ in the paper can therefore be interpreted as a MLP with three input, $64$ hidden and $64$ output units, or as a _fully convolutional network_ with two layers of $64$ filters each.

#### 1.1.3 Solving permutation invariance

As discussed above, we want our network to predict the same class, no matter in which order the points are presented. Can you think of a function (aka a mathematical operation) that produces the same result, independent of the input order (i.e. that is _commutative_)? Turns out, there are many and its super simple. Here is one: $4+3+1=3+1+4$. Addition. Here is another: $2\times8=8\times2$. Multiplication. The authors opted for the $max$ operator, i.e. $max(5, 7, 10)=max(7,5,10)=10$.

This has the added advantage that the network is pushed to reduce each object to its most important features, as is the case when using max-pooling in convolutional networks, which leads to increased robustness to outliers, missing points and additional points, because the result will be the same as long as the most descriptive points remain. Interestingly, those turned out to be the skeletons of the objects.

---