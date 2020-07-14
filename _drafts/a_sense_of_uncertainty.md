---
layout: post
title: A sense of uncertainty
abstract: See what happens when probability theory and deep learning have a little chat. Part 2/3 of my informal mini-series on probabilistic machine learning ("Looking for Lucy" being the first).
category: learning
tags: [probability, Bayesian inference, deep learning]
mathjax: true
time: 9
words: 2501
---

# {{ page.title }}

## What does it mean and why do we need it?

The first question coming to mind if confronted with the concept of a Bayesian Neural Network is, why even bother? _My standard Neural Networks are working just fine, thank you!_ The answer is, that the world itself is inherently uncertain and a run-of-the-mill Neural Net has no idea what it’s talking about when it classifies your dog as a cat with $99.9\\%$ certainty.

When confronted with a difficult problem like “_What did you eat on Monday two weeks ago?_” you will probably preface whatever answer comes to mind with a _“I’m not quite sure but I think…”_ or _“It could have been…”_. A standard Neural Net can’t do this. It’s more like _“She often eats spaghetti, so that’s what it was!”_

**A note for the critical among you:** You might object that even a standard Neural Network returns a score for each class it’s predicting and you might be tempted to treat those numbers as probabilities of being correct, but there are at least two problems:

1. Theoretical: Simply squishing an arbitrary collection of numbers through a Softmax function doesn’t magically produce real probabilities.
2. Practical: It has been observed time-and-again now, that modern deep Neural Networks are overconfident (a notion we will come back to soon) such that the “confidence” expressed by the “probabilities” of the output layer don’t match the networks empirical frequency of being correct. In other words: A prediction of $0.7$ or $70\\%$ for the class `cat` does not translate into $70$ out of $100$ cat images being classified correctly.

The real world is ambiguous and uncertain in all sorts of ways due to extremely complex interactions of a large number of factors (think, e.g., weather forecasting) and because we only ever observe it through some kind of interface: a camera, a microphone, our eyes. Those interfaces, usually called “sensors” in robotics, have their own problems like struggling with low light or transmitting corrupted information. An agent, be it a biological or artificial, must takes those uncertainties into account when operating within such an environment.

## Flavors of uncertainty

Usually, uncertainty is put into two broad categories which makes it easier to think about it and model it. The first, often called _model uncertainty_[^2] is inherent to the model (or agent) and describes its ignorance towards its own stupidity. A standard neural net is maximally ignorant in that it chooses one, most likely way of explaining everything—which translates into one specific set of parameters or weights—and then runs with it.

[^2]: Or _epistemic uncertainty_.

<b style="color: red;">Todo: Add image of simple neural net with weights</b>

This is equivalent to an old person having figured out the answers to all important questions and being impossible to convince otherwise. A Bayesian Neural Network, just as a biological Bayesian (the person), works differently. It considers all possible ways of looking a the problem (within the limited pool of possibilities granted to it during its design) and weighs them by the amount of evidence it has observed for each of those ways. It then integrates them into one coherent explanation. We will see what that looks like in practice a bit later.

<b style="color: red;">Todo: Add image of simple Bayesian neural net</b>

The second type of uncertainty is commonly referred to as _data uncertainty_[^3] and it’s exactly what it sounds like: is the information provided by the data clearly discernible or not? You might think about a fogy night in the forest where you’re trying to convince yourself, that this moving shape is just a branch of a tree swaying in the wind. You can look at it hard and from multiple angles, possibly reducing your uncertainty about the thing (model uncertainty) but you can’t change the fact that it’s night, foggy and your eyes simply aren’t cut for this kind of task (data uncertainty). This also sheds light onto the fact that model uncertainty can be reduced (with more data) but data uncertainty cannot (as it’s inherent to the data).

[^3]: Or _aleatoric uncertainty_.

<b style="color: red;">Todo: Think of a way to visualize both kinds of uncertainty in an image</b>

Finally, both uncertainty flavors can be combined into an overall uncertainty about you decision: the _predictive uncertainty_. This is usually what one refers to when speaking about the topic of uncertainty and it is often simpler to obtain than the former two.

## Modeling uncertainty

Now that we are certain about our need of uncertainty, we need to express it somehow. The only reason a human being doesn’t need a blueprint to do so is, that it has been indirectly hammered in by evolution and experience. In the sciences, this is done through the language of [probability theory](https://hummat.github.io/learning/2020/06/23/looking-for-lucy.html).

Before we can go any further, we need to sharpen up our vocabulary used to refer to specific things. Let's first introduce our main protagonist: The neural network. It's getting a bit more technical now, so feel free to review some of the necessary [background knowledge](#a-some-background) if you're struggling to follow.

### Notation

A neural network is a non-linear mapping from _input_ $\boldsymbol{x}$ to (predicted) _output_ (or target) $\boldsymbol{\hat{y}}=f_W(\boldsymbol{x})$, parameterized by _model parameters_ (or _weights_) $W$, where we assume the true target $\boldsymbol{y}$ was generated from our deterministic function $f$ plus noise $\epsilon$ such that $\boldsymbol{y}=f_W(\boldsymbol{x})+\epsilon$. The entirety of inputs and outputs is our data $$\mathcal{D}=\{(\boldsymbol{x}_i,\boldsymbol{y}_i)\}_{i=1}^N=X,Y$$, i.e we have $N$ pairs of input and output where all the inputs are summarized in $X$ and all the outputs are summarized in $Y$. **Bold** symbols denote vectors while UPPERCASE symbols are matrices.

In our case, the inputs are images and the outputs are vectors of scalars, one for each possible class (or label) the network can predict (e.g. `cat` and `dog`), so our network provides a mapping from a bunch of real numbers (the RGB values of the pixels of the image) to a number of classes. This means we are dealing with a _classification_ rather than a _regression_ problem.

<b style="color: red;">Todo: Image in RGB layers with pixel raster and mapping to output vector</b>

### Becoming Bayesian

So far, everything has been deterministic, which includes the weights $W$ of the neural network. But how do we actually know that those are the best possible weights? Actually, we don't. We hope that they are reasonable by minimizing the loss during training. [As you know](#a-some-background), this is achieved by following the negative gradient (i.e. going in the direction of steepest _decent_) of the loss w.r.t. to the weights, using the _backpropagation_ (i.e. the chain rule from calculus) algorithm to compute this gradient. _Gradient descent optimization_ puts this aptly and succinctly.

{% include figures/loss3d.html %}

What you might not know, if you haven't been exposed to probabilistic machine learning, is, that this is equivalent to _maximum likelihood estimation_ in statistical inference. Imagine you want to [find a friend on a large ship](https://hummat.github.io/learning/2020/06/23/looking-for-lucy.html). You don't know where she is but you think some locations, like the restaurant or the sun deck, are more likely than others, like the rear or the engine room. To quantify your uncertainty, you can use the tools of probability theory and model your world view as a probability distribution. If locations you deem likely are in the middle of the ship and less likely once are at the front and rear, you could, for example, use a normal distribution (aka _Gaussian distribution_ or simply _Gaussian_ in the machine learning community) centered at the middle of the ship to reflect this. What we just did is called _modeling_ of a parameter, the location of your friend, and the probability you assigned to each possible location is called _prior probability_ (or just _prior_), because it is your belief about the world before having observed any evidence (or information, or data) was observed.

{% include figures/ship_1dgauss.html %}

Now imagine someone told you she was seen at the rear of the ship. This is a new piece of information which you should incorporate into your belief in order to maximize your likelihood of finding your friend efficiently. As the explanation already gave away, the probability to obtain this new information _given_ your current belief about the parameter(s) you are trying to estimate is correct, is called _likelihood_.

Both of these terms, _prior_ and _likelihood_, also appear in the probabilistic interpretation of neural network training. The information we base our beliefs about the likelihood of specific parameter configurations on is the data $\mathcal{D}$, while those parameters are the weights $W$. The difference to the deterministic setting is now, that we don't work with a single set of most likely weights $W^\star$, but with a _distribution_ over all possible weight configurations $p(\mathcal{D}\vert W)$, i.e. the likelihood function. We can still find the most likely set of weights though, by looking for the maximum of the likelihood as $W^\star=\arg\max_W p(\mathcal{D}\vert W)$, i.e. _maximum likelihood estimation_.

Why is this identical to gradient descent optimization now? Because, as it turns out, the typical loss functions used in neural network training, which are the cross entropy loss for classification and mean-squared error loss for regression problems, both have a probabilistic interpretation as the _negative log likelihood_, i.e. the negative logarithm of the likelihood! And because the logarithm preserves critical points and we _minimize_ the _negative_ likelihood in gradient descent optimization, this is the same as _maximizing_ the actual likelihood in maximum likelihood estimation.

**Mean-squared error:**

$$
E(W)=\frac{1}{2}(\boldsymbol{y}-\boldsymbol{\hat{y}})^T(\boldsymbol{y}-\boldsymbol{\hat{y}})=-\ln\mathcal{N}\left(\boldsymbol{y}\vert\boldsymbol{\hat{y}},\beta^{-1}I\right)
$$

**Cross entropy:**

$$
E(W)=-\sum_{c=1}^K\boldsymbol{y}_c\ln\boldsymbol{\hat{y}}_c=-\ln\mathrm{Cat}(\boldsymbol{\hat{y}})
$$

As you might have glimpsed from these equations, the mean-squared error can be interpreted as the negative logarithm of an isotropic multivariate normal distribution of the true labels centered around the predictions with precision $\beta$ while the cross entropy has an interpretation as the negative logarithm of a categorical distribution of the predictions.

{% include figures/loss_vs_gauss.html %}

We have discussed the loss-likelihood relation so let's turn to the prior. As the prior encompasses all assumptions about the parameters we want to estimate, almost everything that is known as _regularizers_ in standard machine learning lingo can be cast into this framework. Those are basic things like the choice and design of our model, i.e. using a neural network and giving it a specific number of layers and other architectural decision but also, more explicitly, regularization of possible values we allow the weights to take on, the most common being _weight decay_, aka $L_2$-regularization, where larger values are penalized. Especially this last example, again, has a specific probabilistic interpretation, becoming a Gaussian prior in the probabilistic context.

Introducing a prior into the mix elevates our maximum likelihood estimate to a _maximum a posteriori_ estimate, owing to Bayes' theorem, telling us that the posterior distribution is proportional to the likelihood times the prior, i.e. $p(W\vert\mathcal{D})\propto p(\mathcal{D}\vert W)\cdot p(W)$. So while gradient descent optimization performs maximum likelihood estimation absent any form of regularization, it seamlessly elevates to maximum a posteriori estimation once regularization is introduced where the optimal weights are found by maximizing the posterior instead of the likelihood such that $W^\star=\arg\max_W p(W\vert\mathcal{D})$.

### Bayesian Inference

Bot the maximum a posteriori and the maximum likelihood estimate are so called _point estimates_. This makes sense, because even though we have _modeled_ the weights probabilistically, we don't make full use of the fact by then reducing the likelihood or posterior _distributions_ to a single, most likely value $W^\star$ through maximization.

We can now simply plug those weights into our network and perform _inference_: Making predictions on new, unobserved data. This final step however can also be performed in a proper probabilistic way, through _marginalization_: removing the influence of a random variable by summing (if discrete) or integrating (if continuous) over all of its possible values. Suppose we are given a new datum, e.g. an unobserved image of an animal $\boldsymbol{x}^\star$, for which we would like to predict the class $\boldsymbol{y}^\star$. Using deterministic inference, we would simply use our neural network with trained weights:[^4]

[^4]: Not the $\approx$ because [a neural network doesn't provide actual class _probabilities_](#what-does-it-mean-and-why-do-we-need-it) in general.

$$
\boldsymbol{y}^\star=f_{W^\star}(\boldsymbol{x}^\star)\approx p(\boldsymbol{y}^\star\vert\boldsymbol{x}^\star,W^\star)
$$

In Bayesian inference, we are not using the best weights, instead we are using _all possible_ weights. This means we don't need to keep them around any longer, having extracted all the information we could, so we end up with just $p(\boldsymbol{y}^\star\vert\boldsymbol{x}^\star)$. How can we obtain this? We can start by adding everything else we are given, apart from the new input, namely the rest of the data and our neural network, i.e. the weights. Now we have $p(\boldsymbol{y}^\star\vert\boldsymbol{x}^\star,W,\mathcal{D})$. But the predicted class of the new image doesn't depend on the data we've received so far, so we can split the expression into $p(\boldsymbol{y}^\star\vert\boldsymbol{x}^\star,W)\cdot p(W\vert\mathcal{D})$. The first term is our neural network: Given an image and a set of weights, it can predict class probabilities. The second term is the posterior: Given the data, it tells us the likelihood of all possible weights we might choose. If we want to get rid of the influence of the weights, which, after all, are just an arbitrary choice we've made to model the problem, we can integrate them out:

$$
p(\boldsymbol{y}^\star\vert\boldsymbol{x}^\star)=\int p(\boldsymbol{y}^\star\vert\boldsymbol{x}^\star,W)p(W\vert\mathcal{D})\mathrm{d}W
$$

Both deterministic as well as Bayesian inference finally use the most likely class, i.e. the maximum of $\boldsymbol{y}^\star$, as the prediction.

## Bayesian Neural Networks

* Placing probability distributions on weights

* Ways of estimating the distribution and weights
