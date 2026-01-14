---
layout: post
title: Truly understanding 1x1 convolutions
abstract: There are many explanations out there trying to convince you of the utility of 1x1 convolutions as bottlenecks to reduce computational complexity and replacements for fully connected layers, but they always glanced over some pretty important details in my opinion. Here is the full picture.
tags: [convolution, bottleneck, inception, shared mlp, pointnet]
category: learning
mathjax: True
image: https://assets.hummat.com/images/fc_vs_conv/conv_rgb.png
time: 7
words: 1893
---

# {{ page.title }}

In this short post we will finally understand $1\times1$ convolutions. I say finally, because I've thought many times now that I had understood them only to find a use-case where their application again didn't seem to make sense.

My insufficient understanding stemmed of course from insufficient engagement, for which I'm to blame, but also from deficiencies in the explanations I found. While trying to understand the [PointNet](https://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf) architecture for deep learning on point clouds for example, I was again mystified by the concept of a "shared MLP" (multi-layer perceptron) "sliding" over the input and how it was implemented using convolutions. Because the explanatory content became too long for a simple interlude inside another article, I decided to devote a full post to the subject.

To get started, let's quickly recap how a fully connected layer[^1] would operate on a monochrome (black and white) image. Below (figure 1) I've visualized how a single neuron (a) and an entire fully connected layer (b) operates on such a $28\times28=784$ pixel image. The mathematical operation performed by _each unit_ in this layer is

[^1]: Also often called _dense layer_.

$$y=\sum_{i=1}^{784}w_ix_i+b,$$

where $b$ is an offset, usually called the _bias_ (not shown in the figure). This means we obtain $784$ scalar outputs $y$ from the layer, one from each unit.

<div style="text-align: center">
<figure style="width: 45%; display: inline-block;">
  <img src="https://assets.hummat.com/images/fc_vs_conv/fc_single.png">
  <figcaption style="text-align: left;  line-height: 1.2em;"><b>Fig. 1 (a):</b> A single fully connected neuron, only showing connections (weights $w$) for two inputs (pixel).</figcaption>
</figure>
<figure style="width: 45%; display: inline-block">
  <img src="https://assets.hummat.com/images/fc_vs_conv/fc_full.png">
  <figcaption style="text-align: left; line-height: 1.2em;"><b>Fig. 1 (b):</b> A fully connected layer, only showing four of the $784$ neurons and two of the $784$ inputs each.</figcaption>
</figure>
</div>

Now let's see how we can mimmick these operations using convolutions. In what follows, I'm assuming some basic familiarity with (convolutional) neural networks on your part. If that's not the case, have a look at [the background section](/learning/2020/07/17/a-sense-of-uncertainty.html#some-background) of one of my previous posts or [this excellent explanation](https://cs231n.github.io/convolutional-networks/) from the famous CS231n Stanford course to brush up your knowledge.

To learn from images, we present them, one by one or in batches of multiple images, to a stack of convolutional layers, each consisting of a stack of filters in turn. This exploits the structure of images, where neighboring pixels are assumed to be highly correlated, to reduce the number of parameters compared to a fully connected approach, where each pixel gets its own weight.

A standard convolutional layer is defined by the number of input channels, e.g. the red, green and blue color channels of an image for the input layer, the height and width of the kernels or weight matrices (we have one kernel per input channel) which are convolved with the input channels to give the convolutional neural network its name, and the number of output channels, or feature maps, or filters. A convolutional layer operating on an RGB image with $3\times3$ kernels and 128 filters would therefore be of dimension `input channel`x`kernel height`x`kernel width`x`output channel` i.e. $3\times3\times3\times128$. The first dimension is usually omitted, as it is deemed obvious (i.e. easily inferred from the number of channels from the input) and the output channel are sometimes stated in the first (_"channels first"_) or last (_"channels last"_) position. Below (figure 2) you see a simple convolution on a monochrome (black and white) input image (a) and the conceptually easy to imagine implementation using a "sliding fully connected" network (b).

<div style="text-align: center">
<figure style="width: 45%; display: inline-block;">
  <img src="https://assets.hummat.com/images/fc_vs_conv/conv_slide.png">
  <figcaption style="text-align: left;  line-height: 1.2em;"><b>Fig. 2 (a):</b> A standard convolution of a single filter with one $3\times3$ kernel.<br><br></figcaption>
</figure>
<figure style="width: 45%; display: inline-block">
  <img src="https://assets.hummat.com/images/fc_vs_conv/fc_slide.png">
  <figcaption style="text-align: left; line-height: 1.2em;"><b>Fig. 2 (b):</b> Conceptually simple: Sliding a "fully connected" layer over the image, restricting it to $9$ adjacent inputs.</figcaption>
</figure>
</div>

The idea of sliding a small network over the input, as shown in fig. 2 (b), initially introduced in the [Network in Network paper](https://arxiv.org/pdf/1312.4400.pdf%20http://arxiv.org/abs/1312.4400.pdf), seems quite intuitive. If you were to actually implement this though, you would notice that it's a non-trivial task, because both "sliding" and "only being partially connected" are not part of the standard repertoire of a fully connected layer. Instead, let's try to express a fully connected layer as a convolution, which slides and partially connects natively. To do so, we have two options:

1. Using one filter per input pixel with one large kernel ($28\times28$) per input channel (figure 3).
2. Using one filter per input pixel with one small kernel ($1\times1$) per input channel _and_ pixel (figure 5).

<div style="text-align: center">
<figure style="width: 90%; display: inline-block;">
  <img src="https://assets.hummat.com/images/fc_vs_conv/conv_full2.png">
  <figcaption style="text-align: left;  line-height: 1.2em;"><b>Fig. 3:</b> The operation of a fully connected layer on a two-dimensional input (only looking at spatial dimensions here, not depth, i.e. number of channels) can be described by a convolution with one kernel (weight matrix) per input channel (one) of the same size as the input ($28\times28$ pixel), repeated for each input pixel (i.e. $784$ filter/output channel).</figcaption>
</figure>
</div>


I think the first approach is relatively intuitive. The convolution operation performed by each filter is identical to that stated above for a single fully connected unit, i.e. multiplying a unique scalar weight with each input pixel and summing them up. Consequently, we obtain the same $784$ scalar outputs, one from each filter, perform the same number of operations (multiplications and additions) and have the same number of parameters ($784\times784=614656$ omitting biases).

If we reduce the width and height of our filter kernels to one, we obtain $1\times1$ convolutions. Let's first look at a trivial example (figure 4) with a single filter and three kernels, operating on an RGB image (a) and again the conceptually simple extension to the fully connected approach (b).

<div style="text-align: center">
<figure style="width: 45%; display: inline-block;">
  <img src="https://assets.hummat.com/images/fc_vs_conv/conv_rgb.png">
  <figcaption style="text-align: left;  line-height: 1.2em;"><b>Fig. 4 (a):</b> A standard convolution of a single filter with three $1\times1$ kernels.<br><br></figcaption>
</figure>
<figure style="width: 45%; display: inline-block">
  <img src="https://assets.hummat.com/images/fc_vs_conv/fc_rgb.png">
  <figcaption style="text-align: left; line-height: 1.2em;"><b>Fig. 4 (b):</b> Conceptually simple: Sliding a "fully connected" layer over the image, restricting it to one pixel per channel.</figcaption>
</figure>
</div>

The important thing to note here is, that both approaches produce a _single_ output per spatial dimension (i.e. width and height), because, per convention[^2], convolutions of input channels and kernels from the same filter are summed up, meaning each filter only produces one feature map. In the example above, the red, green and blue pixel values are multiplied by the red green and blue weights and then summed to give a scalar output. This is actually the more common way to employ $1\times1$ convolutions, namely as a way to reduce or increase depth, i.e. the number of output channels and not to mimic fully connected operations. For example, we can put this to use in the final layer of our network to produce one feature map per class by sliding $10$ filter with $1\times1$ kernels over the output feature maps of the previous layer. In the figure below (6), this would produce one feature map for each digit from one to ten, where each "pixel" in the feature map corresponds to the "oneishness" or "twoishness" of each input pixel[^3].

[^2]: This part is often omitted!
[^3]: Interestingly, summing those feature maps over the spatial dimensions would again produce the same result as a fully connected layer, i.e. one scalar value per class, so we could view it as a third way of mimicking fully connected layers with convolutions.

<div style="text-align: center">
<figure style="width: 45%; display: inline-block;">
  <img src="https://assets.hummat.com/images/fc_vs_conv/conv_slide2.png" style="padding: 50px 0 50px 0">
  <figcaption style="text-align: left;  line-height: 1.2em;"><b>Fig. 5 (a):</b> Using $1\times1$ convolutions to produce "class" feature maps with <em>one filter</em> per class.<br><br></figcaption>
</figure>
<figure style="width: 45%; display: inline-block">
  <img src="https://assets.hummat.com/images/fc_vs_conv/fc_slide2.png">
  <figcaption style="text-align: left; line-height: 1.2em;"><b>Fig. 5 (b):</b> The same effect achieved with a fully connected approach, were we to focus on one pixel at a time and capable of sliding the layer across the input.</figcaption>
</figure>
</div>


Armed with the knowledge from the previous paragraph, we can now understand the second approach of transforming a fully connected layer into a convolutional layer. To do so, we first _transform the image into a vector_[^4] by concatenating all of its pixels and then apply _one filter per pixel_ with _one kernel per pixel_, as our image now has as many channels as it had pixels (i.e. it is of shape $1\times1\times784$). Have a look at figure 6 below to take it in visually.

[^4]: Another important information often omitted (or not considered?) in the explanations I found on the subject.

<div style="text-align: center">
<figure style="width: 45%; display: inline-block;">
  <img src="https://assets.hummat.com/images/fc_vs_conv/conv_single1.png" style="padding: 50px 0 50px 0">
  <figcaption style="text-align: left;  line-height: 1.2em;"><b>Fig. 6 (a):</b> A $28\times28$ pixel image transformed into a $1\times1\times784$ vector (left) is convolved with a single filter with $784$ one-by-one kernels (right) producing a single scalar value.</figcaption>
</figure>
<figure style="width: 45%; display: inline-block">
  <img src="https://assets.hummat.com/images/fc_vs_conv/conv_full1.png">
  <figcaption style="text-align: left; line-height: 1.2em;"><b>Fig. 6 (b):</b> The same input image as in (a), but now convolved with $784$ filters, each with $784$ one-by-one kernels, producing $784$ scalar outputs.<br><br></figcaption>
</figure>
</div>

Because each filter produces a single scalar (as input dimensions are summed), we again end up with $784$ outputs. So it's actually not as simple as exchanging the fully connected layer by a $1\times1$ convolution to gain the same functionality. Instead, we need to first reshape the image and then apply as many filter as there are pixel in the input.

That's it! If there are other ways to understand $1\times1$ convolutions, better ways to visualize them or any mistakes, please let me know. I hope this really cleared things up for you as it did for me when writing and visualizing the subject.

---
