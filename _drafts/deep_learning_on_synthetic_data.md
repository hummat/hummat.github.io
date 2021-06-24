---
layout: post
title: Deep Learning on photorealistic synthetic data
abstract: If you work in machine learning or worse, deep learning, you have probably encountered the problem of too few data at least once. For a classification task you might get away with hand-labeling a couple of thousand images and even detection might still be within manual reach if you can convince enough friends to help you. And then you also want to do segmenation. Even if possible, hand-labeling is an incredibly boring, menial task. But what if you could automate it by rendering photorealistic synthetic training data with pixel-perfect annotations for all kinds of scene understanding problems? That's what I would like to showcase in this article.
tags: [synthetic data, photorealism, deep learning]
category: learning
jquery: true
slideshow2: true
time: 0
words: 0
---

# {{ page.title }}

## Introduction

Let me preface this by encouraging you to keep reading regardless of you level of expertise in the field. I think the approach presented here is so general yet intuitive that it can benefit novices and experts alike while being supremely accessible.

So, what is this and why is it exciting? You might be aware of the growing level of realism of computer generated contend in both the games and film industry, to the point where it's sometimes indistinguishable from the real world. If this is completely new to you, I would encourage you to give it a quick search online. You will be amazed by how much of modern movies is actually not real but computer generated to the point where only the actors faces remain (if at all).

Now, if you are reading this, chances are you are neither a cinematographer nor game designer, so why should you care? Here is why I do: At work, I'm partly responsible for making our robots perceive the world. This is mostly done through images from cameras and we use neural networks to extract meaning from them. But neural networks need to be trained and they are not exactly quick learners. This means you need to provide tons of examples of what you want the network to learn before it can do anything useful. The most common tasks a robotic perception system needs to solve are object detection and classification but sometimes we might also need segmentation and pose estimation.

<div style="text-align: center">
<figure style="width: 90%; display: inline-block;">
  <img class="img-swap" src="/images/cup/coco.jpg">
  <figcaption style="text-align: left; line-height: 1.5em;"><b>Instance Segmentation:</b> Every <em>pixel</em> of every <em>instance</em> of each object, e.g. <em>couch</em> or <em>chair</em>, needs to be labeled in every image. The resulting <em>instance segmentation masks</em> can be visualized as semi-transparent overlays (hover over the image to see them). This insane amount of work leads to imperfect results: Only a subset of all visible objects gets labeled, not all instances get labeled (one of the two vases in the bookshelf is missing) or are lumped together (lower rows of books in the shelf), masks are not pixel-perfect (the light blue mask of the arm-chair) and objects get wrong labels (the fireplace is labeled as tv, the armchair in front is also labeled as couch).</figcaption>
</figure>
</div>

How do we get training data for these tasks? Well, depending at where you work and what your budget looks like you might enlist friends, coworkers, students or paid workers online to draw those gorgeous _bounding boxes_ around each object of interest in each image and additionally label them with a class name. For segmentation this becomes a truly daunting task and for pose estimation, you can't even do it by any normal means[^1].

[^1]: One way is to use a robotic arm to move each object into predefined, known poses and store the image together with the pose. This severely restricts the variety of backgrounds and lighting (an important point we will come to later) and the gripper of the robotic arm can occlude important parts of objects.

Apart from fatiguing fellow human beings by forcing them to do such boring work, they also get tired and make mistakes resulting in wrong class labels, too large or small bounding boxes and forgotten objects. You probably see where this is going: What if we could automate this task by generating training data with pixel-perfect annotations in huge quantities? Let's explore the potential and accompanying difficulties of this idea through a running example: _The cup_.

![](/images/cup/cup_photo.jpg)

By the end of this article, we want to be able to detect the occurrence and position of this cup in real photographs (and maybe even do segmentation and pose estimation) without hand-annotating even a single training datum.

## Making some data

Before we can make synthetic training data, we first need to understand what it is. It all starts with a _3D model_ of the object(s) we want to work with. There is a lot of great 3D modeling software out there but we will focus on [_Blender_]() because it is free, open source, cross-platform and, to be honest, simply awesome.

{% include /figures/cup.html %}

I made this cup model you see above in an afternoon (so there is still some work involved) but I'm a complete novice to 3D modeling[^2] and an expert could probably make such a simple model in a few minutes. More complex objects might take more time, but there are already a lot of great 3D models out there you can download for free and even if you start from scratch, you only need to do it once and can reuse it for all kinds of projects and learning tasks.

[^2]: At least in this _artistic_ fashion and not using CAD software as is done in engineering. Depending on the kind and complexity of the model, using CAD software can be a better choice for modeling and you can usually export it in a format which can later be imported into Blender for the rest of the data generation pipeline.

We could now snap an artificial image (i.e. a _render_) of the cup model to get our first datapoint! But wait, you might think, where are the promised automatic annotations like bounding boxes and segmentation masks? For now, simply trust the rendering software (Blender) to know where it places everything, and we will come back to this in a bit.

<div id="slideshow1" class="slideshow-container">
  <div class="mySlides">
    <img src="/images/cup/cup_basic.png" style="width: 100%">
    <div class="text" style="text-align: center; bottom: -35px;"><b>The bare minimum:</b> The rendered cup wihtout colors, textures and surface properties.</div>
  </div>

  <div class="mySlides">
    <img src="/images/cup/cup_color.png" style="width: 100%">
    <div class="text" style="text-align: center; bottom: -35px;"><b>Color and reflections:</b> Correct colors and roughness greatly increase realism.</div>
  </div>

  <div class="mySlides">
    <img src="/images/cup/cup_texture.png" style="width: 100%">
    <div class="text" style="text-align: center; bottom: -35px;"><b>Adding subtle effects:</b> Textures, displacements and surface imperfections.</div>
  </div>

  <div class="mySlides">
    <img src="/images/cup/cup_background.jpg" style="width: 100%">
    <div class="text" style="text-align: center; bottom: -35px;"><b>Final result:</b>The cup rendered on a random background with correct lighting.</div>
  </div>

<a class="prev" onclick="plusSlides(-1, this.parentNode)">&#10094;</a>
<a class="next" onclick="plusSlides(1, this.parentNode)">&#10095;</a>

</div>
<br>

First, let's try to get a few more datapoints. Simply rendering the same image a hundred times won't provide any new information to our neural network so we need to introduce some variety. We could rotate the cup to render it from all sides or rotate the (artificial) _camera_ around it, which, in this setup, would be equivalent. Still, a white background with the object in the center isn't exactly realistic and thus won't generalize to real world applications. The most common way to narrow the gap between simulation and the real world (the _sim2real_ gap) is to render objects in front of randomly selected photographs.

While simple, the result is unconvincing due to differences in lighting: the object will be rendered with artificially placed lights which won't match the direction, intensity and color of the light from the scene captured in the photograph. A slightly better approach is to use _high dynamic range images_ (HDRIs) which store, next to color, additional brightness information in every pixel and cover 360 degrees of viewing angle. Rendering an object inside this sphere allows the utilization of light color, direction and intensity for realistic illumination and reflections.

<div id="slideshow2" class="slideshow-container">
  <div class="mySlides">
    <img src="/images/cup/basic/rgb_0051.jpg" style="width:100%">
  </div>

  <div class="mySlides">
    <img src="/images/cup/basic/rgb_0053.jpg" style="width:100%">
  </div>

  <div class="mySlides">
    <img src="/images/cup/basic/rgb_0054.jpg" style="width:100%">
  </div>

  <div class="mySlides">
    <img src="/images/cup/basic/rgb_0062.jpg" style="width:100%">
  </div>

  <div class="mySlides">
    <img src="/images/cup/basic/rgb_0063.jpg" style="width:100%">
  </div>

  <div class="mySlides">
    <img src="/images/cup/basic/rgb_0064.jpg" style="width:100%">
  </div>

<a class="prev" onclick="plusSlides(-1, this.parentNode)">&#10094;</a>
<a class="next" onclick="plusSlides(1, this.parentNode)">&#10095;</a>

</div>

So far, everything we have done can be achieved through Blender alone. Through the introduction of randomized camera views, things started to get tedious though. While you could add a camera path around the object and invoke renderings at discrete positions on this path to obtain multiple views of you object, the result wouldn't be random. Adding further randomizations like lighting, backgrounds and colors requires manual interaction with the program for each setting which doesn't scale to 10.000 images. But why do we even need randomness?

It's all about generalization: You want your trained model to work under diverse conditions like different viewpoints, varying lighting conditions and (usually) slight variations of object properties like size or color. The best known way so far to achieve this is to present the algorithm with this kind of variability during training. In the real world, this can be challenging, leading to _overfitting_ where the algorithms start to memorize correct answers or start to fit noise rather than the intended signal. Imagine for example an algorithm trained to classify the weather on images. If all rainy days in the training data include people with umbrellas, the algorithm could learn to classify images based on the absence or presence of umbrellas rather than sun or clouds in the sky or dry or wet streets. When deployed, such an algorithm would fail miserably on images with people using umbrellas against the sun or on those depicting bad weather but without any human being.

Now, how can we solve this problem in our synthetic data generation pipeline? Well, you probably guessed it, we write a program to automize everything we've done so far and _much_ more.

## BlenderProc

<div id="slideshow3" class="slideshow-container">
  <div class="mySlides">
    <img src="/images/cup/room/rgb_0000.jpg" style="width:100%">
  </div>

  <div class="mySlides">
    <img src="/images/cup/room/rgb_0001.jpg" style="width:100%">
  </div>

  <div class="mySlides">
    <img src="/images/cup/room/rgb_0002.jpg" style="width:100%">
  </div>

  <div class="mySlides">
    <img src="/images/cup/room/rgb_0003.jpg" style="width:100%">
  </div>

  <div class="mySlides">
    <img src="/images/cup/room/rgb_0004.jpg" style="width:100%">
  </div>

<a class="prev" onclick="plusSlides(-1, this.parentNode)">&#10094;</a>
<a class="next" onclick="plusSlides(1, this.parentNode)">&#10095;</a>

</div>

<div id="slideshow4" class="slideshow-container">
  <div class="mySlides">
    <img src="/images/cup/pose/rgb_0000.jpg" style="width:100%">
  </div>

  <div class="mySlides">
    <img src="/images/cup/pose/rgb_0001.jpg" style="width:100%">
  </div>

  <div class="mySlides">
    <img src="/images/cup/pose/rgb_0002.jpg" style="width:100%">
  </div>

  <div class="mySlides">
    <img src="/images/cup/pose/rgb_0003.jpg" style="width:100%">
  </div>

  <div class="mySlides">
    <img src="/images/cup/pose/rgb_0004.jpg" style="width:100%">
  </div>

<a class="prev" onclick="plusSlides(-1, this.parentNode)">&#10094;</a>
<a class="next" onclick="plusSlides(1, this.parentNode)">&#10095;</a>

</div>

<div id="slideshow5" class="slideshow-container">
  <div class="mySlides">
    <img src="/images/cup/textures/rgb_0000.jpg" style="width:100%">
  </div>

  <div class="mySlides">
    <img src="/images/cup/textures/rgb_0001.jpg" style="width:100%">
  </div>

  <div class="mySlides">
    <img src="/images/cup/textures/rgb_0002.jpg" style="width:100%">
  </div>

  <div class="mySlides">
    <img src="/images/cup/textures/rgb_0003.jpg" style="width:100%">
  </div>

  <div class="mySlides">
    <img src="/images/cup/textures/rgb_0004.jpg" style="width:100%">
  </div>

<a class="prev" onclick="plusSlides(-1, this.parentNode)">&#10094;</a>
<a class="next" onclick="plusSlides(1, this.parentNode)">&#10095;</a>

</div>

<div id="slideshow6" class="slideshow-container">
  <div class="mySlides">
    <img src="/images/cup/clutter/rgb_0000.jpg" style="width:100%">
  </div>

  <div class="mySlides">
    <img src="/images/cup/clutter/rgb_0001.jpg" style="width:100%">
  </div>

  <div class="mySlides">
    <img src="/images/cup/clutter/rgb_0002.jpg" style="width:100%">
  </div>

  <div class="mySlides">
    <img src="/images/cup/clutter/rgb_0003.jpg" style="width:100%">
  </div>

  <div class="mySlides">
    <img src="/images/cup/clutter/rgb_0004.jpg" style="width:100%">
  </div>

<a class="prev" onclick="plusSlides(-1, this.parentNode)">&#10094;</a>
<a class="next" onclick="plusSlides(1, this.parentNode)">&#10095;</a>

</div>

<div id="slideshow7" class="slideshow-container">
  <div class="mySlides">
    <img src="/images/cup/scene/rgb_0000.jpg" style="width:100%">
  </div>

  <div class="mySlides">
    <img src="/images/cup/scene/rgb_0001.jpg" style="width:100%">
  </div>

  <div class="mySlides">
    <img src="/images/cup/scene/rgb_0002.jpg" style="width:100%">
  </div>

  <div class="mySlides">
    <img src="/images/cup/scene/rgb_0003.jpg" style="width:100%">
  </div>

  <div class="mySlides">
    <img src="/images/cup/scene/rgb_0004.jpg" style="width:100%">
  </div>

<a class="prev" onclick="plusSlides(-1, this.parentNode)">&#10094;</a>
<a class="next" onclick="plusSlides(1, this.parentNode)">&#10095;</a>

</div>

<div id="slideshow8" class="slideshow-container">
  <div class="mySlides">
    <img src="/images/cup/real/real0.jpg" style="width:100%">
  </div>

  <div class="mySlides">
    <img src="/images/cup/real/real1.jpg" style="width:100%">
  </div>

  <div class="mySlides">
    <img src="/images/cup/real/real2.jpg" style="width:100%">
  </div>

  <div class="mySlides">
    <img src="/images/cup/real/real3.jpg" style="width:100%">
  </div>

  <div class="mySlides">
    <img src="/images/cup/real/real4.jpg" style="width:100%">
  </div>

  <div class="mySlides">
    <img src="/images/cup/real/real5.jpg" style="width:100%">
  </div>

  <div class="mySlides">
    <img src="/images/cup/real/real6.jpg" style="width:100%">
  </div>
  
  <div class="mySlides">
    <img src="/images/cup/real/real7.jpg" style="width:100%">
  </div>

  <div class="mySlides">
    <img src="/images/cup/real/real8.jpg" style="width:100%">
  </div>

  <div class="mySlides">
    <img src="/images/cup/real/real9.jpg" style="width:100%">
  </div>

  <div class="mySlides">
    <img src="/images/cup/real/real10.jpg" style="width:100%">
  </div>

<a class="prev" onclick="plusSlides(-1, this.parentNode)">&#10094;</a>
<a class="next" onclick="plusSlides(1, this.parentNode)">&#10095;</a>

</div>

<video width="700" controls>
  <source type="video/mp4" src="/images/cup/video.mp4">
</video>

- What is this?: Interface to simplify the use of blenders Python API for data generation
- Adding "real" backgrounds
  - Textures
  - Lighting
  - HDR images
  - Imperfections

## Putting it all together

- Gallery with generated images
- Detection results in the real world (GIF?)
- Wrapping up
