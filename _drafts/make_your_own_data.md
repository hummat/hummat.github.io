---
layout: post
title: Deep Learning on photorealistic synthetic data
abstract: If you work in machine learning or worse, deep learning, you have probably encountered the problem of too few data at least once. For a classification task you might get away with hand-labeling a couple of thousand images and even detection might still be within manual reach if you can convince enough friends to help you. And then you also want to do segmenation. Even if possible, hand-labeling is an incredibly boring, menial task. But what if you could automate it by rendering photorealistic synthetic training data with pixel-perfect annotations for all kinds of scene understanding problems? That's what I would like to showcase in this article.
tags: [synthetic data, photorealism, deep learning]
category: learning
mathjax: false
time: 0
words: 0
---

# {{ page.title }}

## Introduction

Let me preface this by encouraging you to keep reading regardless of you level of expertise in the field. I think the approach presented here is so general yet intuitive that it can benefit novices and experts alike while being supremely accessible.

So, what is this and why is it exciting? You might be aware of the growing level of realism of computer generated contend in both the games and film industry, to the point where it's sometimes indistinguishable from the real world. If this is completely new to you, I would encourage you to give it a quick search online. You will be amazed by how much of modern movies is actually not real but computer generated to the point where only the actors faces remain (if at all).

Now, if you are reading this, chances are you are neither a cinematographer nor game designer, so why should you care? Here is why I do: At work, I'm partly responsible for making our robots perceive the world. This is mostly done through images from cameras and we use neural networks to extract meaning from them. But neural networks need to be trained and they are not exactly quick learners. This means you need to provide tons of examples of what you want the network to learn before it can do anything useful. The most common tasks a robotic perception system needs to solve are object detection and classification but sometimes we might also need segmentation and pose estimation.

Todo: Examples of perception tasks (on hover).

How do we get training data for these tasks? Well, depending at where you work and what your budget looks like you might enlist friends, coworkers, students or paid workers online to draw those gorgeous _bounding boxes_ around each object of interest in each image and additionally label them with a class name. For segmentation this becomes a truly daunting task and for pose estimation, you can't even do it by any normal means[^1].

[^1]: One way is to use a robotic arm to move each object into predefined, known poses and store the image together with the pose. This severely restricts the variety of backgrounds and lighting (an important point we will come to later) and the gripper of the robotic arm can occlude important parts of objects.

Apart from fatiguing fellow human beings by forcing them to do such boring work, they also get tired and make mistakes resulting in wrong class labels, too large or small bounding boxes and forgotten objects. You probably see where this is going: What if we could automate this task by generating training data with pixel-perfect annotations in huge quantities? Let's explore the potential and accompanying difficulties of this idea through a running example: _The cup_.

Todo: Add photograph of cup.

By the end of this article, we want to be able to detect the occurrence and position of this cup in real photographs (and maybe even do segmentation and pose estimation) without hand-annotating even a single training datum.

## Making some data

- Adding texture
- Adding light
- Randomization: Why?
- Adding the camera / different views
- Adding random backgrounds
- How to get even more realistic?

Before we can make synthetic training data, we first need to understand what it is. It all starts with a _3D model_ of the object(s) we want to perceive. There is a lot of great 3D modeling software out there but we will focus on [_Blender_]() because it is free, open source, cross-platform and, to be honest, simply awesome.

Todo: Add 3D visualization of the cup using plotly.

I made this cup model you see above in an afternoon (so there is still some work involved) but I'm a complete novice to 3D modeling[^2] and an expert could probably make such a simple model in a few minutes. More complex objects might take more time, but there are already a lot of great 3D models out there you can download for free and even if you start from scratch, you only need to do it once and can reuse it for all kinds of projects and learning tasks.

[^2]: At least in this _artistic_ fashion and not using CAD software as is done in engineering. Depending on the kind and complexity of the model, using CAD software can be a better choice for modeling and you can usually export it in a format which can later be imported into Blender for the rest of the data generation pipeline.

We could now snap an artificial image (i.e. a _render_) of the cup model to get our first datapoint! But wait, you might think, where is the promised automatic annotation? For now, simply trust the rendering software to know where the objects it is rendering are, and we will come back to this in a bit.

First, let's try to get a few more datapoints. Simply rendering the same image a hundred times won't provide any new information to our neural network so we need to introduce some variety. We could rotate the cup to render it from all sides or rotate the (artificial) _camera_ around it, which, in this setup, would be equivalent. 

## BlenderProc

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
