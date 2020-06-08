---
layout: post
title: Laplace Approximation for Bayesian Deep Learning
abstract: Making a deep neural network Bayesian is a difficult task. For my Masters' thesis I've been using Laplace Approximation to achieve this and developed a plug & play PyTorch implementation I would like to showcase in this post. Let's dive in!
tags: [laplace approximation, Bayesian inference, deep learning]
category: [repository]
gradient: true
github: https://hummat.github.io/curvature
mathjax: true
time: 0
words: 869
---

This article, like every other in the category _repository_, showcases the theory behind and usage of some code I’ve written and made public on GitHub. Still, this post is somewhat special, because the topic is what I’ve been working on during my [Masters’ thesis](https://elib.dlr.de/131938/1/Humt_thesis.pdf), so expect an especially detailed (and long) explanation.

May hope is that, by the end of this article, you get away with a more than superficial understanding of what differentiates a _Bayesian Neural Network_ from a standard _Neural Network_, why Bayesian Neural Networks are great and how we can transform a Neural Network into a Bayesian Neural Network using _Laplace Approximation_. If you additionally found a way to wring some utility from my code I would be truly happy!

Before going into any detail I would like to give a short overview of the topics touched upon ([B](#b-what-to-expect)) in this post as well as those that won’t be covered ([A](#a-some-background)), as it would make this discussion unbearably long (and because I believe there are already vastly superior explanations out there)[^1].

[^1]: Just click on the small black arrows to read more.

Let’s start by the topics I won’t cover but for which I’ll supply some resources so you can brush up your knowledge if needed. If you are like me and long resource lists give you anxiety because you feel obliged to read, watch, understand all of it _before_ you can even start reading the article (which often results in an infinite regression into the depth of the Internet), don’t. Just pick whatever looks interesting or especially unclear or simply start reading the article and come back to the resources if something doesn’t make sense.

## A. Some background

<details>
<summary>Read</summary>

* **Linear algebra & calculus:** Okay, I know, you see this everywhere and for me at least, it   always feels discomforting. What is it supposed to mean anyway? Do I need to know _all_ of linear algebra and calculus to understand anything? And what does _“know”_ mean? That I can solve matrix multiplications, determinants, Eigenvectors and 10th degree derivatives by hand in a few seconds? That I can proof the fundamental equations that underly those fields? I don’t think so.

  Usually, and this is also true here, it just means that you have an _intuitive_ understanding   of what is happening when multiplying a vector and a matrix or what a 2nd order derivative represents. Luckily, this kind of understanding can be obtained conveniently and even enjoyably by watching the following three video series (by one of my YouTube idols [3Blue1Brown](https://www.youtube.com/c/3blue1brown) who we will probably encounter again and again throughout this section and even throughout this blog):

  * [Essence of linear algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
  * [Essence of calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
  * [Multivariate calculus](https://www.youtube.com/playlist?list=PLSQl0a2vh4HC5feHa6Rc5c0wbRTx56nF7)

* **Probability theory:** As you might have expected from the title, where there is Bayes, probability theory can’t be far. Again, an intuitive understanding will suffice to understand what’s going on.

  * [Seeing theory](https://seeing-theory.brown.edu/)
  * [Probability explained](https://www.youtube.com/playlist?list=PLC58778F28211FA19)
  * [Bayes theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM) and its [proof](https://www.youtube.com/watch?v=U_85TaXbeIo) (optional)
  * [Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/) (optional)

* **[Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi):** This is the second ingredient next to probability theory you need to construct a Bayesian Neural Network. 3Blue1Brown one more time.

* **Machine Learning:** Not strictly needed, but so cool that I need to share it. A visual introduction to machine learning: Part [1](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/) and [2](http://www.r2d3.us/visual-intro-to-machine-learning-part-2/)
</details>

## B. What to expect

<details>
<summary>Read</summary>

1. **Bayesian Neural Networks:** A guided tour from standard Neural Networks towards their Bayesian counterparts. I’ll be focusing particularly on the connection between the visual explanation of _“placing a probability distribution on the networks weights”_ (a phrase encountered in most articles on the topic) and what that actually looks like when applied (something that tripped me up quite a bit in the beginning).

2. **Laplace Approximation:** This might be the most interesting part for readers already familiar with the basics. We will look at what it is, why it makes sense to use it and how it can be applied to Neural Networks. There will be some math here but also lots of accompanying visualizations to allow for an intuitive understanding of the presented equations. Where applicable I’ll also contrast this method against other popular approaches, pointing out some advantages and disadvantages. Throughout this section I’ll add some code snippets from the repository to connect theory and application.

3. **Application and results:** The reward for sticking with me till this point: Lot’s of figures visualizing the most interesting results and some analysis. I’ll also provide a complete [small scale example](https://github.com/hummat/curvature/blob/master/curvature/tutorial.ipynb) showing how to use the code so that hopefully you won’t have any problems incorporating it into a personal project, should you be so inclined.
</details>

## 1. A sense of uncertainty

* why uncertainty?
  * The world is inherently uncertain
  * Agents operating in it must take it into account
  * Two types of uncertainty: : Aleatoric (data) and epistemic (model)
* notation
  * Give everything a name
  * input, output, NN, parameters/weights
  * unknown data

* neural nets
  * classification and regression
* placing a probability distribution on a variable: from variable to random variable
* from NN to BNN
  * Bayesian learning

## 2. Being normal around the extreme

* Which distribution? Expressiveness vs computational capacity
* Laplace’s method
* Applying it to NNs to get BNNs

## 3. Into the wild

* Getting the Gaussian
* Go Gaussian, go!
  * calibration
  * ood
  * adversarial