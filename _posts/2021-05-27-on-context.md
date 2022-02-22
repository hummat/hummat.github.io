---
layout: post
title: Context
abstract: A short introduction to the importance of context in machine learning which also serves as an introduction to the upcoming article on attention.
tags: [context, attention]
category: learning
thumbnail: /images/context.png
slow: true
slideshow2: true
jquery: true
mathjax: true
plotly: true
time: 7
words: 1687
---

# {{ page.title }}

Welcome to this little journey where we discover some fundamental concepts in the realm of (machine) learning, namely context and attention. The story is told in two acts: 1. Why and 2. How. We will cover the _why_ and the legacy part of _how_ in this article and then take a closer look at a modern approach in the next one.

## Why? Conceiving context

It all begins with a little entity making up most of the (digital) world around you. It takes many names some calling it _word_, _pixel_ or _point_, but we will simply call it _element_. Our little element is secretive, revealing almost nothing about itself in isolation. In that regard, it is like its sibling in the real world, the atom. Both are atomic[^1]. It has emergent properties though: Throw a couple of thousand of them together and you get a story, an image, a 3D model. What has changed? The Context.

> **Context:** The circumstances that form the setting for an event, statement, or idea, and in terms of which it can be fully understood.

Let's look at a couple of examples. The simplest (an therefore the one we will see most frequently throughout the article) is the word. Try to guess the meaning of the word below, then hover over it with your mouse (or tap on it) to reveal the context:

<img class="img-animate" src="/images/attention/bank.png">

Did you guess the meaning correctly? Or was it the financial institution or place to sit? The point is, of course, that you couldn't have known without the context of the entire sentence, as many words are ambiguous. It doesn't stop there though. Even the sentence is ambiguous if your goal is to determine the book title or author who wrote it. To do so, you might need a paragraph, a page or even an entire chapter of context. In machine learning lingo, such broad context is commonly called a _long-range dependency_. Here is another one. Pay attention to the meaning of the word _it_:

![](/images/attention/it.gif)

Seeing _tired_, we know _it_ must refer to the animal, as roads are seldom so while it's the opposite for _wide_[^2].

Below, there are two more examples of increasing dimensionality (use the little arrows to switch between them). While sentences can be interpreted as one-dimensional sequences of word-elements, an image is a two-dimensional grid of picture-elements (pixels) and a 3D model can be represented by a cloud of point-elements[^3] (or volumetric-elements: voxels). You will notice that you can't discern what is represented by the closeup view of the individual elements but when zooming out (using the "Zoom out" buttons and your mousewheel or fingers) the interpretation becomes trivial.

<br/>
<div id="slideshow1" class="slideshow-container">
  <div class="mySlides fade">
    <div data-include="/figures/image_zoomed.html"></div>
  </div>

  <div class="mySlides fade">
    <div data-include="/figures/happy_buddha.html"></div>
  </div>

<a class="prev" onclick="plusSlides(-1, this.parentNode)">&#10094;</a>
<a class="next" onclick="plusSlides(1, this.parentNode)">&#10095;</a>

</div>
<br/>

Again, context doesn't stop there. To correctly place a pixel as belonging to, say, an eye, you need the surrounding pixels making up the eye. To place the eye as coming from an adult or a child you make use of the information stored in the pixels around the eye. Such inference can potentially go on indefinitely, but it's usually restricted by the size of the depicted scene or the resolution of the image. Okay, you might think, so is more information always better? No.

![](/images/attention/knowledge.gif)

Finding the hidden information in the image above is trivial if the surrounding context is removed (to be precise, it's not the _absence_ of context, as all pixels are still there, but the _contrast_ between signal and noise, percieved as difference between gray and colored pixels). Clearly, it's not a simple as having no context at all or all of it but rather which portion of the provided information we pay _attention_ to.

[^1]: Not really of course, words can be divided into letters, atoms into particles, but let's ignore that.
[^2]: This, and many more of these (deliberately) ambiguous sentences can be found in the _Winograd schema challenge_.
[^3]: Also known as a _point cloud_. Take a look at the previous articles on learning from 3D data for other representations.

## How? Context across dimensions and domains

Now that you are hopefully convinced that context is an important concept across domains, let's start this section off by investigating how researchers have dealt with it prior to the advent of attention. First up are sequence data in the form of written and spoken language. Then we will look at images and 3D data formats in turn.

### Context without attention

For a long time, the predominant method used to model natural language was the _Recurrent Neural Network_ (RNN), first in a basic fully-connected flavor and later using _Long-Short-Term Memory_ (LSTM) and _Gated Recurrent Units_ (GRU). In this paradigm, context is accumulated over time, one element (word) after another, allowing the model to reference potentially indefinitely into the past. Potentially, because in reality it turns out that the memory is rather short and plagued by vanishing and exploding gradients, a problem addressed to some extend by the LSTM and GRU variants. The recurrent nature, while interesting, also requires sequential computation, resulting in slow training and inference. Additionally, the model can't look into the future,[^4] requiring a complicated encoder-decoder design, first aggregating the entire context of the input to provide the decoder with the necessary global context,[^5] a point we will return to in the last section of this article.

<div style="text-align: center">
<figure style="width: 90%; display: inline-block;">
  <img src="/images/attention/rnn_unrolled.png">
  <figcaption style="text-align: left; line-height: 1.2em;"><b>An unrolled RNN:</b> Input elements $\boldsymbol{x}_t$ are processed sequentially while context is retained in the hidden states $\boldsymbol{h}_t$. Taken from [<a href="#ref1">1</a>].</figcaption>
</figure>
</div>

[^4]: Except for bi-directional RNNs which read the sequence from left and right.
[^5]: An example being machine translation: The input sequence (a sentence in English) is first encoded from the first to the last element (word) and then decoded sequentially to produce the translation (the sentence in French).

The shortcomings of RNNs motivated the look into alternatives, one of which was found in a revered companion: the convolution. At first glance, this might seem like a strange choice, considering convolutions as almost synonymous with locality.
However, there are at least two tricks to aggregate long-range dependencies using convolutions. The first one is to simply stack them. Maybe due to their prevalence in image processing, the range covered by a convolution is called its _receptive field_ and through stacking, it can be grown.

<div style="text-align: center">
<figure style="width: 90%; display: inline-block;">
  <img width="400" style="margin-left: auto; margin-right: auto;" src="/images/attention/1d_conv.png">
  <figcaption style="text-align: left; line-height: 1.2em;"><b>Receptive field:</b> The receptive field (red) is the range of context aggregated into the current representation. It increases with the number of stacked convolutional layers. Adapted from [<a href="#ref2">2</a>].</figcaption>
</figure>
</div>

Now, for large inputs (a long text, a high resolution image, a dense point cloud), this simple way of increasing the receptive field size is inefficient, because we need to stack many layers which bloats the model. A more elegant way is to use _strided_ convolutions, where the convolutional kernel is moved more than a single element, or _dilated_ (atrous) convolutions, where the kernel weights are scattered across the input with perceptive holes (French: _trous_) in between. As we might miss important _in between_ information with this paradigm we can again stack multiple such convolutions with varying strides or dilation factors to efficiently cover the entire input.

<div style="text-align: center">
<figure style="width: 90%; display: inline-block;">
  <img src="/images/attention/wavenet.gif">
  <figcaption style="text-align: left; line-height: 1.2em;"><b>WaveNet:</b> Using dilated convolutions, long sequences can be processed efficiently while retaining a large receptive field. Taken from [<a href="#ref3">3</a>].</figcaption>
</figure>
</div>

Moving to the image domain, there is no fundamentally new idea here as vision models still largely rely on convolutions with similar characteristics as introduces above, the only change being the added second dimension.

<div style="text-align: center">
<figure style="width: 90%; display: inline-block;">
  <img src="/images/attention/dilated.gif">
  <figcaption style="text-align: left; line-height: 1.2em;"><b>Dilation in 2D:</b> Concepts like strided and dilated convolutions work identically in one, two or three dimensions. Taken from [<a href="#ref4">4</a>].</figcaption>
</figure>
</div>

Adding a third dimension, things get more interesting again, as the computational complexity of convolutions becomes a major problem. While they can be used successfully, the input usually needs to be downsampled considerably prior to their application. Another approach is to use an element-wise feed-forward neural network[^6]. This approach is extremely efficient, but doesn't consider _any_ context. To resolve this, context aggregation is performed by an additional process like _Farthest Point Sampling_, _k Nearest Neighbor_ search or _Ball Queries_. One exception is the _Graph Neural Network_. As the name implies, it works on graphs as input (either dynamically computed or static ones as found in triangle meshes) and can leverage graph connectivity for context information. I've written an entire mini-series on learning from various 3D data representations which I invite you to check out if the above seems inscrutable.

[^6]: Sometimes referred to as _shared MLP_ (Multi-Layer Perceptron), which in the end boils down to a 1x1 convolution as discussed [here](https://hummat.github.io/learning/2020/10/29/one-by-one-conv.html).

<br/>
<div id="slideshow2" class="slideshow-container">
  <div class="mySlides fade">
    {% if page.slow %}
    <div data-include="/figures/bunny_with_spheres.html"></div>
    {% else %}
    <img src="/images/bunny.png">
    {% endif %}
    <div class="text" style="text-align: left; bottomt: -60px; width: 90%;"><b>Point context:</b> Defining context regions using farthest point sampling and ball queries.</div>
  </div>

  <div class="mySlides fade">
    {% if page.slow %}
    <div data-include="/figures/3d_conv.html"></div>
    {% else %}
    <img src="/images/3d_conv.png">
    {% endif %}
    <div class="text" style="text-align: left; bottom: -60px; width: 90%;"><b>Convolutions in 3D:</b> Adding a dimension drastically increases the computational burden of convolutions, making them cumbersome in the 3D domain.</div>
  </div>

  <div class="mySlides fade">
    {% if page.slow %}
    <div data-include="/figures/graph.html"></div>
    {% else %}
    <img width="220" style="margin-left: auto; margin-right: auto; display: block;" src="/images/mesh.png">
    {% endif %}
    <div class="text" style="text-align: left; bottom: -60px; width: 90%;"><b>Graph context:</b> A mesh can be interpreted as a graph where context is expressed through connectivity.</div>
  </div>

<a class="prev" onclick="plusSlides(-1, this.parentNode)">&#10094;</a>
<a class="next" onclick="plusSlides(1, this.parentNode)">&#10095;</a>

</div>
<br/>

### Taming context with attention

Cliffhanger. See you in the next post.

## References

|                        |                                                                                                                                                                 |
| :--------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [1<a name="ref1"></a>] | [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs)                                                                        |
| [2<a name="ref2"></a>] | [UNetGAN](https://arxiv.org/abs/2010.15521)                                                                                                                     |
| [3<a name="ref3"></a>] | [WaveNet: A generative model for raw audio](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio)                                               |
| [4<a name="ref4"></a>] | [Review: Dilated Residual Networks](https://towardsdatascience.com/review-drn-dilated-residual-networks-image-classification-semantic-segmentation-d527e1a8fb5) |

---
