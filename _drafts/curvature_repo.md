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

1. **Uncertainty:** Because probabilistic, or Bayesian, machine learning is not (yet) part of the fields 101, I’d like to begin this discussion by answering the following three questions: What is uncertainty? Why do we need it? How to put it to use?
2. **Bayesian Neural Networks:** A guided tour from standard Neural Networks towards their Bayesian counterparts. I’ll be focusing particularly on the connection between the visual explanation of _“placing a probability distribution on the networks weights”_ (a phrase encountered in most articles on the topic) and what that actually looks like when applied (something that tripped me up quite a bit in the beginning).
3. **Laplace Approximation:** This might be the most interesting part for readers already familiar with the basics. We will look at what it is, why it makes sense to use it and how it can be applied to Neural Networks. There will be some math here but also lots of accompanying visualizations to allow for an intuitive understanding of the presented equations. Where applicable I’ll also contrast this method against other popular approaches, pointing out some advantages and disadvantages. Throughout this section I’ll add some code snippets from the repository to connect theory and application.
4. **Application and results:** The reward for sticking with me till this point: Lot’s of figures visualizing the most interesting results and some analysis. I’ll also provide a complete [small scale example](https://github.com/hummat/curvature/blob/master/curvature/tutorial.ipynb) showing how to use the code so that hopefully you won’t have any problems incorporating it into a personal project, should you be so inclined.
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

Usually, uncertainty is put into two broad categories which makes it easier to think about it and model it. The first, often called _model uncertainty_ is inherent to the model (or agent) and describes its ignorance towards its own stupidity. A standard neural net is maximally ignorant in that it chooses one, most likely way of explaining everything—which translates into one specific set of parameters or weights—and then runs with it. This is equivalent to an old person having figured out the answers to all important questions and being impossible to convince otherwise. A Bayesian Neural Network, just as a biological Bayesian (the person), works differently. It considers all possible ways of looking a the problem (within the limited pool of possibilities granted to it during its design) and weighs them by the amount of evidence it has observed for each of those ways. It then integrates them into one coherent explanation. We will see what that looks like in practice a bit later.

The second type of uncertainty is commonly referred to as _“data uncertainty”_ and it’s exactly what it sounds like: is the information provided by the data clearly discernible or not? You might think about a fogy night in the forest where you’re trying to convince yourself, that this moving shape is just a branch of a tree swaying in the wind. You can look at it hard and form multiple angles, possibly reducing your uncertainty about the thing (model uncertainty) but you can’t change the fact that it’s night, foggy and your eyes simply aren’t cut for this kind of task (data uncertainty). This also sheds light onto the fact that model uncertainty can be reduced (with more data) but data uncertainty cannot (as it’s inherent to the data).

Finally, both uncertainty flavors can be combined into an overall uncertainty about you decision: the _predictive uncertainty_. This is usually what one refers to when speaking about the topic and often simpler to obtain than the previous two.

### Modeling uncertainty

Now that we are certain about our need of uncertainty, we need to express it somehow. The only reason a human being doesn’t need a blueprint to do so is, that it has been indirectly hammered in by evolution and experience. Fortunately, we can bridge the gap from biological to artificial through this (admittedly slightly contrived) example:

#### Lost on a ship

You are on large ship looking for your friend Lucy. You already have some suspicion as to her whereabouts:

1. You expect her to be on the ship and not in the water (though possible, you deem it unlikely)
2. It is 1 am, a time where she likes to eat lunch, so there is a good chance she’s in the ships restaurant in the middle of the ship.

In probability theory, we call such beliefs for which we don’t have seen any evidence yet _prior beliefs_ or simply _priors_. We’ve also implicitly established the parameter or variable we are trying to estimate: Lucy’s location. Let’s call it $\ell$. We can then write “_I think Lucy is more likely to be on the ship than in the water_” as $p(\ell=\mathrm{ship})=0.9$  and $p(\ell=\mathrm{water})=0.1$

This reads: _“The probability of  $\ell$ taking the value `ship` is $90\\%$“_ (and $10\\%$ for `water` respectively). Two more things to note here:

1. We’ve just defined a _probability distribution_ of $\ell$, written $p(\ell)$, which maps each possible state of $\ell$ (`ship`, `water`) to a discrete probability. In doing so, we promoted $\ell$ from mere variable to _random_ variable. Why random? Because it doesn’t take only one deterministic value like $5$, but different values with different probability. And because Lucy can only either be on the ship or not, the probabilities of these two states need to sum to 1 (or $100\\%$).

2. She also can’t be a little bit on the ship and a little bit in the water (well, technically she probably could, but let’s keep it simple) which is why we call it a _discrete_ probability distribution rather than a _continuous_ one, where everything in between certain values is also possible.

Let’s define such a continuous probability distribution for Lucy’s location _on the ship_. We could of course start enumerating all locations (`restaurant`, `toilet`, `her room`, `sun deck`, …), but because she could be _everywhere_ on the ship,  it’s tedious at best and impossible otherwise. We want to keep using $\ell$ to distinguish between `ship` and `water` so let’s use $e$ to talk about Lucy’s _exact_ location on the ship.

What do you think this means: $p(e|\ell=\mathrm{ship})$? It’s the probability distribution of Lucy’s exact location _given_ she is on the ship. We call this a _conditional_ probability distribution, because it depends on Lucy being on the ship. Therefore $p(e|\ell=\mathrm{water})=0$. No point in establishing an exact location if she’s in the ocean. Let’s say the ship is $50$ meters long, so Lucy could be anywhere between $e=0$ and $e=50$. So what’s $p(e=27|\ell=\mathrm{ship})$? Maybe surprisingly, it’s $0$. Why? Because for a continuous random variable, no exact values exist.

To understand this, think about forecasting the temperature for the next day. What’s the probability that it will be between $-40^\circ C$ and $+40^\circ C$? Probably close to $100\\%$ (though never _actually_ $100\\%$). Between $0^\circ C$ and $30^\circ C$? Still quite high, say $70\\%$ (the exact numbers don’t matter here). Between $25$ and $27$? Okay, that’s much, much less likely, let’s go with $2\\%$. $25.1$ and $25.2$? Almost zero. $25.345363$ and $25.345364$? You get the point. So when talking about probabilities in the continuous case, always think in intervals. If we belief that Lucy is somewhere in the middle of the ship _and_ that she in fact is on the ship, we could write it like this: $p(20\leq e<30|\ell=\mathrm{ship})=0.8$.

What if we want to take into account our prior beliefs about whether she’s on the ship or not? We multiply it! If we’re $90\\%$ certain that she’s on the ship and $80\\%$ certain that she’s in the middle of the ship ($20\leq e < 30$) _if she’s on it_, than our overall belief for this scenario is $0.9\cdot 0.8 = 0.72$.[^2]

[^2]: Always take prior probabilities into account when dealing with conditionals, otherwise you will fall pray to the _base rate fallacy_.

> **Choosing a team:** There is another interesting observation to be made here: Those probabilities we’ve chosen are _beliefs_ about how the world is rather than facts. We therefore make use of _Bayesian statistics_ instead of _Frequentist statistics_, where probabilities are seen as properties of the world to be measured. A fair coin for example has a $50\\%$ chance of landing either `head` or `tail` and you can find out about this fact through observation of repeated experiments. We’ll come back to the distinction between those two views on probabilities later when talking about _calibration_.

Let’s now try to visualize all of this.

This is what’s meant when we talk about _modeling_ in statistics. 

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