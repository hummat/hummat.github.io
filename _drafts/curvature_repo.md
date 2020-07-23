---
layout: post
title: Laplace Approximation for Bayesian Deep Learning
abstract: Making a deep neural network Bayesian is a difficult task. For my Masters' thesis I've been using Laplace Approximation to achieve this and developed a plug & play PyTorch implementation I would like to showcase in this post. This is the final part of my informal 3 part mini-series on probabilistic machine learning, part 1 and 2 being "Looking for Lucy" and "A sense of uncertainty".
category: repository
tags: [laplace approximation, Bayesian inference, deep learning]
gradient: true
github: https://hummat.github.io/curvature
mathjax: true
time: 12
words: 3150
---

Complex problems require complex solutions. While this is not always true, it certainly is often enough to inspire the search into automated problem solving. That’s where machine learning comes into play, which, in the best case, allows us to throw a bunch of data at an algorithm and to obtain a solution to our problem in return.

With the advent of deep learning, this trend has been getting a mighty boost, allowing us to (begin to) solve much harder and thereby more interesting problems, like autonomous driving or automated medical diagnosis. In those areas, where human well being is on the line,  the algorithms opinions  need to be trustworthy and comprehensible. While predictions should be correct, it is even more important to know when they are not, so that one can take appropriate countermeasures, as the real world is messy and often unpredictable, so striving to not make any mistakes ever is probably in vain.

A tragic example of this was the accident of a semi-autonomous vehicle crashing into a white semi truck, mistaking it for a cloud [source](). Here, an algorithm knowing when it doesn’t know, could have warned the driver to take over. Knowing when you don’t know comes down to placing appropriate confidence in your predictions which is true for humans and algorithms alike. We can also frame this problem from the opposite direction, when talking about the level of confidence as low and high uncertainty.

The mathematical language of uncertainty is probability theory, for which I provided a tiny introduction in my [first article](https://hummat.github.io/learning/2020/06/23/looking-for-lucy.html) of the informal three-part mini-series on probabilistic machine learning of which this article is the third and thereby final part. In the [previous article](https://hummat.github.io/learning/2020/07/17/a-sense-of-uncertainty.html), the second part, I told the story of deep learning and probability theory becoming friends, but we didn’t really get to see how these theoretical insights could be applied in praxis, which is the topic of this article.

More precisely, we will be looking at a practical yet powerful way to model a deep neural network probabilistically, thereby transforming it into a Bayesian neural network using a technique called _Laplace Approximation_. Once we are done with that, we will see how to make use of the newly obtained Bayesian superpowers to solve, or at least mitigate, some problems arising from poor calibration, i.e. being over- or underconfident in ones predictions. I’ll show some results I’ve obtained during the work on my [Masters’ thesis](https://elib.dlr.de/131938/1/thesis.pdf) through a plethora of interesting figures and visualizations.

Incidentally, the article will also serve as a tutorial to the [GitHub repository](https://hummat.github.io/curvature/) featuring the code used to obtain all the presented results. If you are uncertain (no pun intended) about your level of background knowledge (of which you certainly need some), head over to the [background section of part two](https://hummat.github.io/learning/2020/07/17/a-sense-of-uncertainty.html#some-background) to verify or brush up your knowledge, or just press on and see if you get stuck at some point.


## 1. Being normal around the extreme

In life as in probabilistic machine learning we are confronted with many open questions, two of which are:

1. How to obtain an accurate assessment of my ignorance (estimating the data likelihood and/or parameter posterior in probability theory speak)?
2. How to make predictions and take decisions properly grounded in this assessment (integrating out or marginalizing the parameters of our model)?

The answer to the first question is: By modeling it probabilistically. We have already seen this in part one, where we modeled our assumptions and the obtained evidence about the location of a friend on a large ship and I’ve also hinted at it in the second part, where we observed, that the (low dimensional) weight posterior distribution of a neural network has a distinctly Gaussian appearance.

Let’s quickly revisit this second fact here. We already know that the loss function used to train the network has a probabilistic interpretation as the negative logarithm of the data likelihood: $E(W)=-\log p(\mathcal{D}\vert W)$[^1].

[^1]: Please refer to [part two](https://hummat.github.io/learning/2020/07/17/a-sense-of-uncertainty.html#notation) for an introduction of the notation.

* Which distribution? Expressiveness vs computational capacity
* Laplace’s method
* Applying it to NNs to get BNNs

## 2. Into the wild

* Getting the Gaussian
* Go Gaussian, go!
  * calibration
  * ood
  * adversarial

---