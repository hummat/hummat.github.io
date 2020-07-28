---
layout: post
title: Laplace Approximation for Bayesian Deep Learning
abstract: Making a deep neural network Bayesian is a difficult task. For my Masters' thesis I've been using Laplace Approximation to achieve this and developed a plug & play PyTorch implementation I would like to showcase in this post. This is the final part of my informal 3 part mini-series on probabilistic machine learning, part 1 and 2 being "Looking for Lucy" and "A sense of uncertainty".
category: repository
tags: [laplace approximation, Bayesian inference, deep learning]
thumbnail: /images/loss_vs_gauss.png
gradient: true
github: https://hummat.github.io/curvature
mathjax: true
time: 8
words: 2171
---

Complex problems require complex solutions. While this is not always true, it certainly is often enough to inspire the search into automated problem solving. That’s where machine learning comes into play, which, in the best case, allows us to throw a bunch of data at an algorithm and to obtain a solution to our problem in return.

With the advent of deep learning, this trend has been getting a mighty boost, allowing us to (begin to) solve much harder and thereby more interesting problems, like autonomous driving or automated medical diagnosis. In those areas, where human well being is on the line,  the algorithms opinions  need to be trustworthy and comprehensible. While predictions should be correct, it is even more important to know when they are not, so that one can take appropriate countermeasures, as the real world is messy and often unpredictable, so striving to not make any mistakes ever is probably in vain.

A tragic example of this was the accident of a semi-autonomous vehicle crashing into a white semi truck, mistaking it for a cloud [source](). Here, an algorithm knowing when it doesn’t know, could have warned the driver to take over. Knowing when you don’t know comes down to placing appropriate confidence in your predictions which is true for humans and algorithms alike. We can also frame this problem from the opposite direction, when talking about the level of confidence as low and high uncertainty.

The mathematical language of uncertainty is probability theory, for which I provided a tiny introduction in my [first article](https://hummat.github.io/learning/2020/06/23/looking-for-lucy.html) of the informal three-part mini-series on probabilistic machine learning of which this article is the third and thereby final part. In the [previous article](https://hummat.github.io/learning/2020/07/17/a-sense-of-uncertainty.html), the second part, I told the story of deep learning and probability theory becoming friends, but we didn’t really get to see how these theoretical insights could be applied in practice, which is the topic of this article.

More precisely, we will be looking at a practical yet powerful way to model a deep neural network probabilistically, thereby transforming it into a Bayesian neural network using a technique called _Laplace Approximation_. Once we are done with that, we will see how to make use of the newly obtained Bayesian superpowers to solve, or at least mitigate, some problems arising from poor calibration, i.e. being over- or underconfident in ones predictions. I’ll show some results I’ve obtained during the work on my [Masters’ thesis](https://elib.dlr.de/131938/1/thesis.pdf) through a plethora of interesting figures and visualizations.

Incidentally, the article will also serve as a tutorial to the [GitHub repository](https://hummat.github.io/curvature/) featuring the code used to obtain all the presented results. If you are uncertain (no pun intended) about your level of background knowledge (of which you certainly need some), head over to the [background section of part two](https://hummat.github.io/learning/2020/07/17/a-sense-of-uncertainty.html#some-background) to verify or brush up your knowledge, or just press on and see if you get stuck at some point.

## 1. Being normal around the extreme

In life as in probabilistic machine learning we are confronted with many open questions, two of which are:

1. How to obtain an accurate assessment of my ignorance (estimating the data likelihood and/or parameter posterior in probability theory speak)?
2. How to make predictions and take decisions properly grounded in this assessment (integrating out or marginalizing the parameters of our model)?

The answer to the first question is: By modeling it probabilistically. We have already seen this in part one, where we modeled our assumptions and the obtained evidence about the location of a friend on a large ship and I’ve also hinted at it in the second part, where we observed, that the (low dimensional) weight posterior distribution of a neural network has a distinctly Gaussian appearance.

Let’s quickly revisit this second fact here. We already know that the loss function used to train the network has a probabilistic interpretation as the negative logarithm of the _likelihood_: $E(W)=-\ln p(\mathcal{D}\vert W)$[^1]. When using some form of regularization to prevent overfitting ([which we almost always do](https://hummat.github.io/learning/2020/07/17/a-sense-of-uncertainty.html#becoming-bayesian)), it becomes the _prior_ $p(W)$ in the probabilistic framework, and we start working with the _posterior_ instead of the likelihood thanks to _Bayes’ Theorem_: $p(W\vert\mathcal{D})\propto p(\mathcal{D}\vert W)p(W)$. In other words, the unregularized loss corresponds to the negative log likelihood while a regularized loss corresponds to the negative log posterior. This posterior is what we need to perform _Bayesian inference_, i.e. to make predictions on new data, like an unseen image of an animal we want to classify. How can we obtain it?

We know that we need to model it probabilistically, i.e. we need to find a probability distribution which captures the important properties of the true, underlying, latent distribution of our model parameters (the weights). Probability distributions come in all kinds of flavors, and we need to strike a balance between _expressiveness_, which determines how well we can model the true distribution, and _practicality_, i.e. how mathematically and computationally feasible our choice is.

### Landscapes of loss

To get an intuitive understanding, let's actually look at a (low dimensional) representation of the likelihood. The likelihood function $L(W)=p(\mathcal{D}\vert W)$ is, as the name suggests, just a function and we can evaluate it at different inputs $W$[^2]. We can, for example, evaluate it at $W^\star$, the weights obtained after training the network. This should yield the highest likelihood and conversely the lowest loss. We can then explore the space around this minimum by taking small steps in one or two (randomly chosen) directions, evaluating $L(W+\alpha W_R)$ or $L(W+\alpha W_{R1}+\beta W_{R2})$ along the way. $W_{R1,2}$ are random vectors of the same size as $W$ and $\alpha$ and $\beta$ are the step sizes. This is what you see below, though we start by visualizing $-\log L(W)$, the loss $E(W)$, first and then move on to the likelihood which is $\exp(-E(W))$.

[^1]: Please refer to [part two](https://hummat.github.io/learning/2020/07/17/a-sense-of-uncertainty.html#notation) for an introduction of the notation.
[^2]: Make no mistake, even though the likelihood is the probability distribution over the data (given the weights), we are still trying to find the _weights_ that best explain the data and not the other way round.

{% include /figures/loss/mobilenet_v2_cifar10_loss_3d.html %}

The so called _loss landscape_ you see above has a few distinctive features[^3]. The most important is, that it is basically a steep valley, at least close to the minimum. This is mostly what allowed our optimizer to find the minimum in the first place, by following the negative gradient in the direction of steepest descent. You can also see that further away from the minimum, small hills begin to appear. The more chaotic the landscape and the further away from the minimum we start our optimization, the harder it will generally be to find the minimum. Now try to think of a probability distribution that could potentially model the shape of this loss landscape. If you can't quite see it yet, let's look at the quantity we are actually interested in modeling: the likelihood. All we need to do is to take the exponential of the negative loss, which is what you see on the left below.

[^3]: [Here](https://github.com/hummat/hummat.github.io/tree/master/_includes/figures/loss) are some more examples, if you are interested as well as the [paper](https://arxiv.org/abs/1712.09913) describing the approach.

{% include /figures/loss/mobilenet_v2_cifar10_loss_vs_gauss.html %}

And the thing on the right that looks very similar? This is a two-dimensional multivariate normal distribution, aka a Gaussian! Note how the negative exponent has smoothed out those small hills we saw in the previous plot, as large (negative) values get mapped to near zero. In this example, I have estimated the correct parameters for the Gaussian, i.e. the mean and variance, through a least-squares fitting approach. However, so far we have been working with a two-dimensional representation of the loss for visualization purposes while in reality, it has as many dimensions as our network has weights! For the popular _ResNet50_ architecture for example, that's more than 25 million! Clearly we need a better approach to estimating the Gaussian parameters, mostly because we can't gather sufficient loss samples in 25 million dimensional space[^4].

[^4]: Have a look at [the curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality).

A bit off topic but still interesting: See the lines below the loss landscape? Those are _contour lines_, like you would see on a hiking map, and we can use these to compare the loss landscape to the _accuracy landscape_, which, after all, is the quantity we are actually interested in when confronted with a classification problem. Click on the dropdown menu to select loss and accuracy contour plots and their comparison.

{% include /figures/loss/mobilenet_v2_cifar10_loss_acc_2d.html %}

It gets a bit crowded, but what we can make out is, that, at least for low loss/high accuracy regions, the loss contour levels align remarkably well with specific accuracy levels, meaning the loss is a good predictor of classification accuracy. Nice!

### Getting the Gaussian

Back to the topic: How can we estimate the shape of a function in general? Exactly, we can use _Taylor expansion_. Here is what that looks like:

$$
\ln p(W\vert\mathcal{D})\approx\ln p(W^\star\vert\mathcal{D})-\frac{1}{2}(W-W^\star)^TH(W-W^\star)
$$

Let's unpack this. On the right we see the posterior we want to estimate. We've kept the logarithm when transforming the (negative log) likelihood into the posterior using Bayes theorem but dropped the minus. We expand our Taylor series around the maximum a posteriori estimate $\ln p(W^\star\vert\mathcal{D})$, drop the first order term, because at the extremum, the gradient vanishes, and are left with the second order term. The only unknown quantity here is $H$, the (parameter) _Hessian_ of the (regularized) loss, i.e. of the negative log likelihood or posterior. The Hessian, being the matrix of second order derivatives, describes the curvature of our loss landscape. But wait, didn't we want to model the loss as a Gaussian? Exactly, but see what happens when we take the exponential:

$$
p(W\vert\mathcal{D})\approx\ p(W^\star\vert\mathcal{D})\exp\left(-\frac{1}{2}(W-W^\star)^TH(W-W^\star)\right)\approx\mathcal{N}\left(W^\star,H^{-1}\right)
$$

Boom, the right hand side becomes the (unnormalized[^5]) _probability density function_ of a multivariate normal distribution!

[^5]: In practice, we don't care much about the normalization factor, because under most conditions, the posterior distribution is asymptotically normally distributed as the number of data points goes to infinity.

Okay, so far so good, how to compute the Hessian then? In theory, we could take the gradient of the gradient to obtain it, but in practice this is computationally infeasible for large networks. As it turns out, and this is probably the hardest pill to swallow in this entire article, as I won't be able to go into any detail why this is so, there is an _identical_ quantity[^6] which we _can_ compute: The _Fisher information matrix_ F. It is defined as follows:

$$
F=\mathbb{E}\left[\nabla_W\ln p(\mathcal{D}\vert W)\nabla_W\ln p(\mathcal{D}\vert W)^T\right]
$$

In other words, it is the expected value[^7] of the outer product of the negative gradient of the loss w.r.t. the weights. In one sense, this is easy to compute, because we already have the required gradients from network training, but another problem persists: the size. Just like the Hessian, the Fisher of a deep neural network is of size `number of weights x number of weights` (which we can also write as $\vert W\vert\times\vert W \vert$) which is prohibitively large to compute, store and invert.

[^7]: The expectation is taken w.r.t. the output distribution of the network. If the data distribution is used instead, one obtains the _empirical Fisher_ which doesn't come with the same equality properties to the Hessian compared to the _"true"_ Fisher.
[^6]: This is only the case for networks that use piece-wise linear activation functions (like ReLU) and exponential family loss functions (like least-squares or cross-entropy).

### KFC

There are several ways to shrink the size of our curvature matrix[^8] of which the simplest is to chop it into layer-sized bits. Instead of one gigantic matrix, we end up with $L$ smaller matrices of size $\vert W_\ell\vert\times\vert W_\ell\vert$ where $L$ is the number of layers in our network and $\ell=\{1,2,3,...,L\}$.

{% include /figures/hessian.html %}

[^8]: I'll be using this term instead of Hessian or Fisher (or _Generalized Gauss-Newton_ for that matter), because for our purposes they are equivalent and can all be interpreted as the curvature of the function they represent.

In the figure above you can immediately see the immense reduction in size by comparing the total area of the red squares to that of the initial one. Conceptually, this simplification says, that we assume the layers of the network to be independent from one another. This becomes clearer if you think about the _inverse_ curvature instead, representing the covariance matrix of the multivariate Gaussian we want to estimate, where the off-diagonal terms correspond to the covariances while the diagonal terms are the variances. However, this is still not good if enough.

This brings us to the next simplification, where we also toss out the covariances _within_ each layer, so that we are left with only the diagonal elements of the initial matrix. The first simplification step is referred to as _block-wise_ approximation while the second is called _diagonal_ approximation.

But what if we want to keep some of the covariances? Let's first think about why this could be helpful. Have a look at the two-dimensional Gaussian below. By changing the diagonal values of the $2\times2$ covariance matrix, we can change the spread, or variance, in $x$ and $y$ direction. But what if the network posterior we are trying to model places probability mass somewhere between those axes? For that, we need to _rotate_ the distribution by changing the _covariance_, which you can try out by using the third slider.

{% include /figures/gaussian_covariance.html %}

Now, as I mentioned before, we cannot simply keep the covariances, as the resulting matrices, even using the block-wise approximation, would still be too unwieldy. What we can do though, is an additional decomposition of each curvature block into two smaller matrices called the _Kronecker factors_ using the _Kronecker product_. The mathematical definition is

$$
A\otimes B = \begin{bmatrix}a_{11}B & ... & a_{1n}B\\\vdots & \ddots & \vdots\\a_{m1}B & ... & a_{mn}B\end{bmatrix}
$$

but it's easier to think about it visually. For example, if $A$ is a $2\times2$ matrix, we can color $a_{11}$ to $a_{22}$ with a different color and then, each of the colored squares is multiplied with every of the four gray squares representing $B$ and placed in the corresponding corner of the resulting matrix.

{% include /figures/kronecker_product.html %}

Even for this toy example you can see that we have reduced the $4\times4$ matrix with $16$ elements to two $2\times2$ matrices with $4+4=8$ elements. Once we use the curvature matrices as covariance matrices, we will have to invert them though, but very conveniently, the inverse of the Kronecker product is the same as the Kronecker product of the inverse factors: $(A\otimes B)^{-1}=A^{-1}\otimes B^{-1}$![^9]

[^9]: Unfortunately, this equality can not generally be maintained in expectation s.t. $\mathbb{E}[A\otimes B]\neq\mathbb{E}[A]\otimes\mathbb{E}[B]$.

This final approximation is called _Kronecker factored curvature_, KFC, and yields a substantially better approximation to the entire curvature matrix compared to a simple diagonal approximation at only a moderate increase in memory requirements.

### Appearing more confident than you are

There is one last obstacle we need to clear before we can use our newly obtained curvature approximation. While in theory a matrix computed by an outer product like the Fisher should always be invertible, in reality this might not necessarily be the case for numerical reasons. But there are two more reasons why we might want to alter those matrices:

1. Our approximations, while necessary to render the problem tractable, could have introduced an unwarranted amount of uncertainty in some directions. 
2. The idea to approximate the weight posterior of our network by a multivariate Gaussian distribution could be flawed, which could happen if the true posterior is not bell shaped or only so in certain directions but not in others. If you have another look at the comparison of the exponential negative loss and the Gaussian above, you can already see that they are only similar, but not identical.

To combat these problems, we can use regularization. In deep learning, the most well known form of regularization is _weight decay_ aka $L_2$-regularization. In the previous article, we have already seen that it has a probabilistic interpretation as a Gaussian prior. This can be easily extended to our case, by adding a multiple of the identity matrix to the curvature matrix. Here is why: Through the introduction of a prior, we are now dealing with the posterior instead of the likelihood. So far, we have been computing the Fisher of the log likelihood, i.e. $F[\ln p(\mathcal{D}\vert W)]$, as [explained before](#getting-the-gaussian).

If we now want to compute the Fisher of the log posterior, $F[\ln p(W\vert\mathcal{D})]$, we can make use of Bayes’ theorem and write

$$
F[\ln p(W\vert \mathcal{D})]\propto F[\ln p(\mathcal{D}\vert W)+\ln p(W)]=F[\ln p(\mathcal{D}\vert W)]+F[\ln p(W)]
$$

Due to the logarithm, the prior is added instead of multiplied and we end up with a sum of the known curvature of the log likelihood (the first term) and the curvature of the log prior, which we will replace by an isotropic Gaussian, i.e. a Gaussian with identical variance in all dimensions and zero covariance. To get rid of the proportionality factor, we can simply add a multiplicative constant an arrive at

$$
\hat{F}=NF+\tau I
$$



### Integration with Monte Carlo

## 2. Into the wild

* Go Gaussian, go!
  * calibration
  * ood
  * adversarial

---