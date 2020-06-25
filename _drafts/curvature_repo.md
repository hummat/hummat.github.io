---
layout: post
title: Laplace Approximation for Bayesian Deep Learning
abstract: Making a deep neural network Bayesian is a difficult task. For my Masters' thesis I've been using Laplace Approximation to achieve this and developed a plug & play PyTorch implementation I would like to showcase in this post. Let's dive in!
category: repository
tags: [laplace approximation, Bayesian inference, deep learning]
gradient: true
github: https://hummat.github.io/curvature
mathjax: true
time: 0
words: 869
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

## 1. A sense of uncertainty

### What does it mean and why do we need it?

The first question coming to mind if confronted with the concept of a Bayesian Neural Network is, why even bother? _My standard Neural Networks are working just fine, thank you!_ The answer is, that the world itself is inherently uncertain and a run-of-the-mill Neural Net has no idea what it’s talking about when it classifies your dog as a cat with $99.9\\%$ certainty.

When confronted with a difficult problem like “_What did you eat on Monday two weeks ago?_” you will probably preface whatever answer comes to mind with a _“I’m not quite sure but I think…”_ or _“It could have been…”_. A standard Neural Net can’t do this. It’s more like _“She often eats spaghetti, so that’s what it was!”_

**A note for the critical among you:** You might object that even a standard Neural Network returns a score for each class it’s predicting and you might be tempted to treat those numbers as probabilities of being correct, but there are at least two problems:

1. Theoretical: Simply squishing an arbitrary collection of numbers through a Softmax function doesn’t magically produce real probabilities.
2. Practical: It has been observed time-and-again now, that modern deep Neural Networks are overconfident (a notion we will come back to soon) such that the “confidence” expressed by the “probabilities” of the output layer don’t match the networks empirical frequency of being correct. In other words: A prediction of $0.7$ or $70\\%$ for the class `cat` does not translate into $70$ out of $100$ cat images being classified correctly.

The real world is ambiguous and uncertain in all sorts of ways due to extremely complex interactions of a large number of factors (think, e.g., weather forecasting) and because we only ever observe it through some kind of interface: a camera, a microphone, our eyes. Those interfaces, usually called “sensors” in robotics, have their own problems like struggling with low light or transmitting corrupted information. An agent, be it a biological or artificial, must takes those uncertainties into account when operating within such an environment.

### Flavors of uncertainty

Usually, uncertainty is put into two broad categories which makes it easier to think about it and model it. The first, often called _model uncertainty_[^2] is inherent to the model (or agent) and describes its ignorance towards its own stupidity. A standard neural net is maximally ignorant in that it chooses one, most likely way of explaining everything—which translates into one specific set of parameters or weights—and then runs with it.

[^2]: Or _epistemic uncertainty_.

<b style="color: red;">Todo: Add image of simple neural net with weights</b>

This is equivalent to an old person having figured out the answers to all important questions and being impossible to convince otherwise. A Bayesian Neural Network, just as a biological Bayesian (the person), works differently. It considers all possible ways of looking a the problem (within the limited pool of possibilities granted to it during its design) and weighs them by the amount of evidence it has observed for each of those ways. It then integrates them into one coherent explanation. We will see what that looks like in practice a bit later.

<b style="color: red;">Todo: Add image of simple Bayesian neural net </b>

The second type of uncertainty is commonly referred to as _data uncertainty_[^3] and it’s exactly what it sounds like: is the information provided by the data clearly discernible or not? You might think about a fogy night in the forest where you’re trying to convince yourself, that this moving shape is just a branch of a tree swaying in the wind. You can look at it hard and from multiple angles, possibly reducing your uncertainty about the thing (model uncertainty) but you can’t change the fact that it’s night, foggy and your eyes simply aren’t cut for this kind of task (data uncertainty). This also sheds light onto the fact that model uncertainty can be reduced (with more data) but data uncertainty cannot (as it’s inherent to the data).

[^3]: Or _aleatoric uncertainty_.

<b style="color: red;">Todo: Think of a way to visualize both kinds of uncertainty in an image</b>

Finally, both uncertainty flavors can be combined into an overall uncertainty about you decision: the _predictive uncertainty_. This is usually what one refers to when speaking about the topic of uncertainty and it is often simpler to obtain than the former two.

### Modeling uncertainty

Now that we are certain about our need of uncertainty, we need to express it somehow. The only reason a human being doesn’t need a blueprint to do so is, that it has been indirectly hammered in by evolution and experience. In the sciences, this is done through the language of [probability theory](https://hummat.github.io/learning/2020/06/23/looking-for-lucy.html).

Before we can go any further, we need to sharpen up our vocabulary used to refer to specific things. Let's first introduce our main protagonist: The neural network. It's getting a bit more technical now, so feel free to review some of the necessary [background knowledge](#a-some-background) if you're struggling to follow.

**Notation:** A neural network is a non-linear mapping from _input_ $\boldsymbol{x}$ to (predicted) _output_ (or target) $\boldsymbol{\hat{y}}=f_W(\boldsymbol{x})$, parameterized by _model parameters_ (or _weights_) $W$, where we assume the true target $\boldsymbol{y}$ was generated from our deterministic function $f$ plus noise $\epsilon$ such that $\boldsymbol{y}=f_W(\boldsymbol{x})+\epsilon$. The entirety of inputs and outputs is our data $$\mathcal{D}=\{(\boldsymbol{x}_i,\boldsymbol{y}_i)\}_{i=1}^N=X,Y$$, i.e we have $N$ pairs of input and output where all the inputs are summarized in $X$ and all the outputs are summarized in $Y$. **Bold** symbols denote vectors while UPPERCASE symbols are matrices.

In our case, the inputs are images and the outputs are vectors of scalars, one for each possible class (or label) the network can predict (e.g. `cat` and `dog`), so our network provides a mapping from a bunch of real numbers (the RGB values of the pixels of the image) to a number of classes. This means we are dealing with a _classification_ rather than a _regression_ problem.

<b style="color: red;">Todo: Image in RGB layers with pixel raster and mapping to output vector</b>

So far, everything has been deterministic, which includes the weights $W$ of the neural network.

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

---