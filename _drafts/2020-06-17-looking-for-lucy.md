---
layout: post
title: Looking for Lucy
abstract: My own take on explaining some fundamentals of probability theory, intended as a primer for probabilistic machine learning.
tags: [probability, statistics]
category: learning
mathjax: true
time: 2
words: 838
---

# {{ page.title }}

You are on large ship looking for your friend Lucy[^1]. You already have some suspicion as to her whereabouts:

[^1]: I know, this is not the ship I had in mind, but it’s the best I could find, so just imagine something pretty and logical.

1. You expect her to be on the ship and not in the water (though possible, you deem it unlikely)
2. It is 1 pm, a time where she likes to eat lunch, so there is a good chance she’s in the ships restaurant in the middle of the ship.

{% include figures/ship.html %}

In probability theory, we call such beliefs for which we don’t have seen any evidence yet _prior beliefs_ or simply _priors_. We’ve also implicitly established the parameter or variable we are trying to estimate: Lucy’s location. Let’s call it $\ell$. We can then write “_I think Lucy is more likely to be on the ship than in the water_” as $p(\ell=\mathrm{ship})=0.9$  and $p(\ell=\mathrm{water})=0.1$

This reads: _“The probability of  $\ell$ taking the value `ship` is $90\\%$“_ (and $10\\%$ for `water` respectively). Two more things to note here:

1. We’ve just defined a _probability distribution_ of $\ell$, written $p(\ell)$, which maps each possible state of $\ell$ (`ship`, `water`) to a discrete probability. In doing so, we promoted $\ell$ from mere variable to _random_ variable. Why random? Because it doesn’t take only one deterministic value like $5$, but different values with different probability. And because Lucy can only either be on the ship or not, the probabilities of these two states need to sum to 1 (or $100\\%$).

2. She also can’t be a little bit on the ship and a little bit in the water (well, technically she probably could, but let’s keep it simple) which is why we call it a _discrete_ probability distribution rather than a _continuous_ one, where everything in between certain values is also possible.

Let’s define such a continuous probability distribution for Lucy’s location _on the ship_. We could of course start enumerating all locations (`restaurant`, `toilet`, `her room`, `sun deck`, …), but because she could be _everywhere_ on the ship,  it’s tedious at best and impossible otherwise. We want to keep using $\ell$ to distinguish between `ship` and `water` so let’s use $e$ to talk about Lucy’s _exact_ location on the ship.

What do you think this means: $p(e\|\ell=\mathrm{ship})$? It’s the probability distribution of Lucy’s exact location _given_ she is on the ship. We call this a _conditional_ probability distribution, because it depends on Lucy being on the ship. Therefore $p(e\|\ell=\mathrm{water})=0$. No point in establishing an exact location if she’s in the ocean. Let’s say the ship is $50$ meters long, so Lucy could be anywhere between $e=0$ and $e=50$. So what’s $p(e=27\|\ell=\mathrm{ship})$? Maybe surprisingly, it’s $0$. Why? Because for a continuous random variable, no exact values exist.

To understand this, think about forecasting the temperature for the next day. What’s the probability that it will be between $-40^\circ C$ and $+40^\circ C$? Probably close to $100\\%$ (though never _actually_ $100\\%$). Between $0^\circ C$ and $30^\circ C$? Still quite high, say $70\\%$ (the exact numbers don’t matter here). Between $25$ and $27$? Okay, that’s much, much less likely, let’s go with $2\\%$. $25.1$ and $25.2$? Almost zero. $25.345363$ and $25.345364$? You get the point. So when talking about probabilities in the continuous case, always think in intervals. If we belief that Lucy is somewhere in the middle of the ship _and_ that she in fact is on the ship, we could write it like this: $p(20\leq e<30\|\ell=\mathrm{ship})=0.8$.

{% include figures/ship_1dgauss.html %}

What if we want to take into account our prior beliefs about whether she’s on the ship or not? We multiply it! If we’re $90\\%$ certain that she’s on the ship and $80\\%$ certain that she’s in the middle of the ship ($20\leq e < 30$) _if she’s on it_, than our overall belief for this scenario is $0.9\cdot 0.8 = 0.72$.[^2]

[^2]: Always take prior probabilities into account when dealing with conditionals, otherwise you will fall pray to the _base rate fallacy_.

> **Choosing a team:** There is another interesting observation to be made here: Those probabilities we’ve chosen are _beliefs_ about how the world is rather than facts. We therefore make use of _Bayesian statistics_ instead of _Frequentist statistics_, where probabilities are seen as properties of the world to be measured. A fair coin for example has a $50\\%$ chance of landing either `head` or `tail` and you can find out about this fact through observation of repeated experiments. We’ll come back to the distinction between those two views on probabilities later when talking about _calibration_.

Now there is actually a better way to represent our two beliefs $e$ and $\ell$, namely in a unified manner using a two dimensional probability distribution! This it what it looks like from above:

{% include figures/ship_2dgauss.html %}

Yellow values are likely areas while purple values are unlikely. The lines connect coordinates of equal density.

The code for the visualizations is available [here]() and you can play with it here: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/hummat/hummat.github.io/master?filepath=%2Fnotebooks%2Flooking-for-lucy.ipynb)

---