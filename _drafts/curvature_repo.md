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

This article, like every other in the category _repository_, showcases the theory behind and usage of some code I’ve written and made public on GitHub. Still, this post is somewhat special, because the topic is what I’ve been working on during my [Masters’ thesis](https://elib.dlr.de/131938/1/Humt_thesis.pdf), so expect an especially detailed (and long) explanation.

May hope is that, by the end of this article, you get away with a more than superficial understanding of what differentiates a _Bayesian Neural Network_ from a standard _Neural Network_, why Bayesian Neural Networks are great and how we can transform a Neural Network into a Bayesian Neural Network using _Laplace Approximation_. If you additionally found a way to wring some utility from my code I would be truly happy!

Before going into any detail I would like to give a short overview of the topics touched upon in this post ([B](#b-what-to-expect)) as well as those that won’t be covered ([A](#a-some-background)), as it would make this discussion unbearably long (and because I believe there are already vastly superior explanations out there)[^1].

[^1]: Just click on the small black arrows to read more.

Let’s start by the topics I won’t cover but for which I’ll supply some resources so you can brush up your knowledge if needed. If you are like me and long resource lists give you anxiety because you feel obliged to read, watch, understand all of it _before_ you can even start reading the article (which often results in an infinite regression into the depth of the Internet), don’t. Just pick whatever looks interesting or especially unclear or simply start reading the article and come back to the resources if something doesn’t make sense.

## A. Some background

<details>
<summary>Read</summary>
<ol>
<li><p><b>Linear algebra & calculus:</b> Okay, I know, you see this everywhere and for me at least, it always feels discomforting. What is it supposed to mean anyway? Do I need to know <i>all</i> of linear algebra and calculus to understand anything? And what does <i>“know”</i> mean? That I can solve matrix multiplications, determinants, Eigenvectors and 10th degree derivatives by hand in a few seconds? That I can proof the fundamental equations that underly those fields? I don’t think so.</p>
<p>Usually, and this is also true here, it just means that you have an _intuitive_ understanding   of what is happening when multiplying a vector and a matrix or what a 2nd order derivative represents. Luckily, this kind of understanding can be obtained conveniently and even enjoyably by watching the following three video series (by one of my YouTube idols <a href="https://www.youtube.com/c/3blue1brown">3Blue1Brown</a> who we will probably encounter again and again throughout this section and even throughout this blog):</p></li>
<ul>
<li><a href="https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab">Essence of linear algebra</a></li>
<li><a href="https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr">Essence of calculus</a></li>
<li><a href="https://www.youtube.com/playlist?list=PLSQl0a2vh4HC5feHa6Rc5c0wbRTx56nF7">Multivariate calculus</a></li>
</ul>
<li><b>Probability theory:</b> As you might have expected from the title, where there is Bayes, probability theory can’t be far. Again, an intuitive understanding will suffice to understand what’s going on. Consider having a look at <a href="https://hummat.github.io/learning/2020/06/23/looking-for-lucy.html">my article</a> on the topic, which is intented specifically as a primer to probabilistic machine learning.</li>
<ul>
<li><a href="https://seeing-theory.brown.edu/">Seeing theory</a></li>
<li><a href="https://www.youtube.com/playlist?list=PLC58778F28211FA19">Probability explained</a></li>
<li><a href="https://www.youtube.com/watch?v=HZGCoVF3YvM">Bayes theorem</a> and its <a href="https://www.youtube.com/watch?v=U_85TaXbeIo">proof</a> (optional)</li>
<li><a href="https://colah.github.io/posts/2015-09-Visual-Information/">Visual Information Theory</a> (optional)</li>
</ul>
<li><a href="https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi"><b>Neural Networks</b></a>: This is the second ingredient next to probability theory you need to construct a Bayesian Neural Network. 3Blue1Brown one more time.</li>
<li><b>Machine Learning:</b> Not strictly needed, but so cool that I need to share it. A visual introduction to machine learning: Part <a href="http://www.r2d3.us/visual-intro-to-machine-learning-part-1/">1</a> and <a href="http://www.r2d3.us/visual-intro-to-machine-learning-part-2/">2</a></li>
</ol>
</details>


## B. What to expect

<details>
<summary>Read</summary>
<ol>
<li><b>Uncertainty:</b> Because probabilistic, or Bayesian, machine learning is not (yet) part of the fields 101, I’d like to begin this discussion by answering the following three questions: What is uncertainty? Why do we need it? How to put it to use?</li>
<li><b>Bayesian Neural Networks:</b> A guided tour from standard Neural Networks towards their Bayesian counterparts. I’ll be focusing particularly on the connection between the visual explanation of <i>“placing a probability distribution on the networks weights”</i> (a phrase encountered in most articles on the topic) and what that actually looks like when applied (something that tripped me up quite a bit in the beginning).</li>
<li><b>Laplace Approximation:</b> This might be the most interesting part for readers already familiar with the basics. We will look at what it is, why it makes sense to use it and how it can be applied to Neural Networks. There will be some math here but also lots of accompanying visualizations to allow for an intuitive understanding of the presented equations. Where applicable I’ll also contrast this method against other popular approaches, pointing out some advantages and disadvantages. Throughout this section I’ll add some code snippets from the repository to connect theory and application.</li>
<li><b>Application and results:</b> The reward for sticking with me till this point: Lot’s of figures visualizing the most interesting results and some analysis. I’ll also provide a complete <a href="https://github.com/hummat/curvature/blob/master/curvature/tutorial.ipynb">small scale example</a> showing how to use the code so that hopefully you won’t have any problems incorporating it into a personal project, should you be so inclined.</li>
</ol>
</details>

## 3. Being normal around the extreme

* Which distribution? Expressiveness vs computational capacity
* Laplace’s method
* Applying it to NNs to get BNNs

## 4. Into the wild

* Getting the Gaussian
* Go Gaussian, go!
  * calibration
  * ood
  * adversarial

---