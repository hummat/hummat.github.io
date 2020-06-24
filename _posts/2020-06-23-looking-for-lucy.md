---
layout: post
title: Looking for Lucy
abstract: My own take on explaining some fundamentals of probability theory, intended as a primer for probabilistic machine learning. We have a look at joint, conditional and marginal probabilities, continuous and discrete as well as multivariate distributions, independence and Bayes' Theorem.
thumbnail: /images/ship_static.png
tags: [probability, statistics]
category: learning
mathjax: true 
words: 3657
time: 14 
update: 2020-06-24
---

# {{ page.title }}

You are on large ship looking for your friend Lucy. You already have some suspicion as to her whereabouts:

[^1]: I know, this is not the ship I had in mind, but it’s the best I could find, so just imagine something pretty and logical.

1. You expect her to be on the ship and not in the water (though possible, you deem it unlikely)
2. It is 1 pm, a time where she likes to eat lunch, so there is a good chance she’s in the ships restaurant in the middle of the ship[^1].

{% include figures/ship.html %}


### Probabilities and Priors

In probability theory, we call such beliefs for which we don’t have seen any evidence yet _prior beliefs_ or simply _priors_. We’ve also implicitly established the parameter or variable we are trying to estimate: Lucy’s location. Let’s call it $\ell$. We can then write “_I think Lucy is more likely to be on the ship than in the water_” as $p(\ell=\mathrm{ship})=0.99$  and $p(\ell=\mathrm{water})=0.01$

This reads: _“The probability of  $\ell$ taking the value `ship` is $99\\%$“_ (and $1\\%$ for `water` respectively). We can often drop the variable name whenever it takes on a specific value, like `ship` , so $p(\ell=\mathrm{ship})$ simply becomes $p(\mathrm{ship})$, to reduce clutter. Two more things to note here:

1. We’ve just defined a _probability distribution_ of $\ell$, written $p(\ell)$, which maps each possible state of $\ell$ (`ship`, and `water`) to a discrete probability ($99\\%$ and $1\\%$). In doing so, we promoted $\ell$ from mere variable to _random_ variable. Why random? Because it’s value is not deterministic like $x=2+2$, but probabilistic, determined by some underlying random (or random seeming) cause. And because Lucy can only either be on the ship or not, the probabilities of these two states need to sum to 1 (or $100\\%$).
2. She also can’t be a little bit on the ship and a little bit in the water (well, technically she probably could, but let’s keep it simple) which is why we call it a _discrete_ probability distribution rather than a _continuous_ one, where everything in between certain values is also possible.

{% include figures/ship_bars.html %}

### Continuous, discrete, conditional

Let’s define such a continuous probability distribution for Lucy’s location _on the ship_. We could of course start enumerating all locations (`restaurant`, `toilet`, `her room`, `sun deck`, …), but because she could be _everywhere_ on the ship,  it’s tedious at best and impossible otherwise. We want to keep using $\ell$ to distinguish between `ship` and `water` so let’s use $e$ to talk about Lucy’s _exact_ location on the ship.

What do you think this means: $p(e\vert \mathrm{ship})$? It’s the probability distribution of Lucy’s exact location _given_ she’s on the ship. The little bar $\vert$ means _given_ or _if_ as in: _”The probability that it will rain if there are clouds”_ $p(\mathrm{rain}\vert \mathrm{clouds})$. We call this a _conditional_ probability distribution, because it depends on Lucy being on the ship. Therefore $p(e\vert \mathrm{water})=0$. No point in establishing an exact location if she’s in the ocean. Let’s say the ship is $50$ meters long, so Lucy could be anywhere between $e=0$ and $e=50$. So what’s $p(e=27\vert \mathrm{ship})$? Maybe surprisingly, it’s $0$. Why? Because for a continuous random variable, no exact values exist.

**Interluding intervals:** To understand this, think about forecasting the temperature for the next day. What’s the probability that it will be between $-40^\circ C$ and $+40^\circ C$? Probably close to $100\\%$ (though never _actually_ $100\\%$). Between $0^\circ C$ and $30^\circ C$? Still quite high, say $70\\%$ (the exact numbers don’t matter here). Between $25$ and $27$? Okay, that’s much, much less likely, let’s go with $2\\%$. $25.1$ and $25.2$? Almost zero. $25.345363$ and $25.345364$? You get the point. So when talking about probabilities in the continuous case, always think in intervals.

If we belief that Lucy is somewhere in the middle of the ship _if_ she in fact is on it, we could write it like this, $p(20\leq e\leq30\vert \mathrm{ship})=0.8$ but because this looks daunting lets call it $p(\mathrm{middle}\vert \mathrm{ship})$ instead.

{% include figures/ship_1dgauss.html %}

The red line is called a _probability density function_ and it describes our continuous probability distribution $p(e\vert\mathrm{ship})$. If the area under the curve between $e=20$ and $e=30$ is Lucys probability to be in the middle of the ship, what’s on the vertical axis then, you might think? It’s where the _density_ comes into play. Like with a physical object, where the mass is determined by its volume and density[^2], so is _probability mass_ determined by its area (in 2D) or volume (in 3D)[^3] and its density on the vertical axis.

If you are familiar with sums and integrals, it might be helpful to look a it this way: In the discrete case, where you simply enumerate all possible locations and attach your belief to each of them, you sum them up to get an overall estimate which can be written like this: $p(\mathrm{restaurant}\cap\mathrm{room})=p(\mathrm{restaurant})+p(\mathrm{room})$ where the flipped U means _or_. Imagine now you discretize the ship into smaller and smaller parts. In the limit, you have covered every micrometer of the ship through an infinite amount of discrete probabilities which is exactly how you can estimate the integral of a function, i.e. the area under the curve.

[^2]: Provided it is made out of one material with the same density everywhere.
[^3]: Or _“hypervolume”_ in 4D and above.

What if we want to take into account our prior beliefs about whether she’s on the ship or not? We multiply! If we’re $90\\%$ certain that she’s on the ship *and* $80\\%$ certain that she’s in the middle of it ($20\leq e < 30$) _if she’s on it_, than our overall belief for this scenario is $0.9\cdot 0.8 = 0.72$.[^4] We call this a _joint probability distribution_ because it expresses our beliefs about two quantities at the same time: That Lucy is on the ship _and_ in the middle of it. Using the quantities introduced earlier we can write it as:

$$p(\mathrm{middle},\mathrm{ship})=p(\mathrm{middle}\vert\mathrm{ship})\cdot p(\mathrm{ship})$$

[^4]: Always take prior probabilities into account when dealing with conditionals, otherwise you will fall pray to the _base rate fallacy_.

See the little comma? That’s all there is to it. Why is this useful? Because often you only have information (or beliefs) about the individual statements but not about both of them together, or the the other way round. For example, what’s the probability that Lucy is in the ships restaurant _and_ eating pizza? That’s kind of hard to reason about. It’s easier to think about her being in the restaurant, which you deem likely, say $70\\%$, and that she’s eating pizza _if_ shes indeed in the restaurant. Say there are two different meals, `pizza` and `spaghetti` and you know Lucy has a slight preference for the former, so $p(\mathrm{pizza})=0.6$ (and therefore $p(\mathrm{spaghetti})=0.4$ as those are the only two options). Now you can say:

$$\begin{aligned}p(\mathrm{restaurant},\mathrm{pizza})&=p(\mathrm{pizza}\vert\mathrm{restaurant})\cdot p(\mathrm{restaurant})\\&=0.6\cdot0.7=0.42\end{aligned}$$

### Independence

In our example, Lucy can’t be in the middle of the ship if she’s in the ocean. Therefore, both statements depend on each other, i.e. having a belief about one influences the other. That’s not always the case though. Saturn is in line with Venus. What can you derive from this knowledge for your personal life? Nothing. Those statements are _independent_, so your probability of being happy $p(\mathrm{happy})$ stays the same, regardless of what Saturn (S) and Venus (V) are up to. This also means we are allowed to do the following:

$$\begin{aligned}p(\mathrm{happy},S\leftrightarrow V)&=p(\mathrm{happy}\vert S\leftrightarrow V)\cdot p(S\leftrightarrow V)\\&=p(\mathrm{happy})\cdot p(S\leftrightarrow V)\end{aligned}$$

The important part is the transition from the first to the second line. There is no conditional probability involved[^5], which is very useful, e.g. in machine learning, as it makes many calculations a lot easier.

[^5]: There is another form of independence though, called _conditional independence_, involving conditional probabilities.

> **Choosing a team:** There is another interesting observation to be made here: Those probabilities we’ve chosen are, for the most part, _beliefs_ about how the world is. We therefore make use of _Bayesian statistics_ instead of _Frequentist statistics_, where probabilities are exclusively seen as intrinsic properties of the world to be measured. A fair coin for example has a $50\\%$ chance of landing either `head` or `tail` and you can find out about this fact through the observation of repeated experiments. We’ll come back to the distinction between those two views on probabilities in future posts when talking about _calibration_.

### More dimensions and variance

Now there is actually a better way to represent our beliefs of Lucys exact location on the ship using a two dimensional probability distribution! Instead of only saying where we expect her to be from back to front, i.e. _rear_ to _bow_, we can now also express our belief about her position from left to right, i.e. _port_ to _starboard_. Let’s call them $e_x$ for rear-bow and $e_y$ for port-starboard position.

Because the ship is longer than wide, there are more possible locations for Lucy to be in that direction.[^6] This can clearly be seen in the _spread_ of the distribution, being more stretched out in $e_x$ direction. This spread is also called the _variance_ of the probability distribution. The variance directly translate into our _uncertainty_ about Lucys location. This it what it looks like from above:

[^6]: If we had another, albeit less reasonable, belief about Lucys location, the distribution could of course look very different, e.g. wider in $e_y$ than in $e_x$ direction. You can think about what this would mean.

{% include figures/ship_2dgauss.html %}

Yellow signifies likely areas while purple values are unlikely.[^7] The lines connect coordinates of equal density, just as the lines on a map connect coordinates of equal altitude. Try rotating the figure to get a better understanding.

[^7]: To be precise: areas of low probability density.

Another side effect of our new 2D distribution is, that we now simultaneously express our belief about Lucy being in the ocean, so we can do away with our additional variable $\ell$! You can also think about what it means that there is more probability mass near the ship than further away from it.[^8]

[^8]: It means that we think it’s more likely that Lucy is still somewhere near the ship if she has fallen into the ocean instead of far away!

### Bayes’ Theorem

The final story I’d like to tell is this one: Suppose you ask another passenger if he has seen a hungry looking woman recently and he tells you that, while he couldn’t tell if she was hungry, he did speak to a women called Lucy at the rear of the ship! What a coincidence.  Such information is called _evidence_ as it tells us something about the parameters we want to model and estimate, namely Lucys location. Let's give this particular piece of information a name: $I$.

How should you deal with the new information? Intuitively you might think it’s a settled case. You are looking for Lucy and there is a Lucy at the rear of the ship. This however would only be true, if you were $100\\%$ certain that the Lucy in question is in fact your friend.

Instead you need to ask the question: _If_ this new information were true, how should it influence my belief? We can’t answer this question right away, so let’s first ask another question: How likely is it to obtain this new information _if_ my current belief about the world is true? In our case, how likely is it that someone might have encountered a Lucy at the rear of the ship (evidence or information) if the Lucy I’m looking for is in the restaurant in the middle of the ship (prior or current belief)? The quantity we are talking about is a special kind of conditional probability and is called _likelihood_. We’ll soon see how it’s used, but first we need to estimate it.

Let’s start by visualizing the total space of possibilities by a square with side length 1:

{% include figures/square.html %}

Why? Because we can use it to visualize probabilities by the area they take up in the square. Half the square: $50\\%$, a quarter: $25\\%$.

**The Prior:** Now there are two possibilities: Either your friend Lucy is in the restaurant or not. The former is a quantity we’ve already estimated, which is our joint belief that she is on the ship _and_ in the middle of it ($72\\%$).

Let’s make this our new _prior_, because it was our belief before we obtained the new information from the other passenger. We’ll call it $p(\mathrm{middle})=0.72$ and put it in our possibility square, by taking up $72\\%$ of it’s width and the entire height:

{% include figures/prior.html %}

What’s the remaining area? It’s the probability (or, more precisely, how likely we think it is) that Lucy is _not_ in the middle of the ship which is $1-0.72=0.28$, i.e. $28\\%$.

**The Likelihood:** You conveniently know that there are 100 people on the ship and overheard a conversation about another Lucy who’s on the ship. How likely is it that she’s at the back? You don’t know anything about her and you assume, for simplicity, that all 100 people are spread out equally around the ship. We already established the ships length to be 50 meters and let’s say the rear is 5 meters long. If the ship has approximately equal width everywhere, the rear makes up $10\%$ of the entire ship. Therefore, you would expect around $10\\%$ of all people to be there. As there are 100 people on board, that’s also conveniently $10\\%$ of those, so the proportion of the area of the ship is the same as the probability to encounter any one specific person there. As there are two Lucys, you would have a $10\\%$ chance to meet each of them, so $2\cdot0.1=0.2$ or $0.1+0.1=0.2$, i.e. $20\\%$, to meet at least one.

{% include figures/ship_locations.html %}

But wait, you’re convinced that your friend Lucy is in the restaurant, so in fact, according to your belief, there is at most _one_ Lucy left strolling around. This means that the probability to meet someone with this name at the rear _if_ your friend is at the restaurant is only half of what we estimated, so only $10\\%$. We call this quantity—the probability to see the evidence given your hypothesis is true—_likelihood_ and write it as $p(I\vert\mathrm{middle})=0.1$. If we want to add this quantity to our probability square, we need to put it in the space where our hypothesis is true, which is inside the green area on the left:

{% include figures/likelihood.html %}

**Evidence:** You might worry, even though you’d be surprised, that Lucy didn’t go to the restaurant. What’s the probability then that someone encountered a Lucy at the rear of the ship? You might be tempted to use our $20\\%$ estimate from before, after all, there are now two Lucys running around, right? Not quite. Because you now assume that she is _not_ in the middle of the ship, there is now a higher probability to encounter her elsewhere (e.g. at the back) compared to the other Lucy (and in fact compared to all other people on board). How much higher? We defined the middle as the area between 20 and 30 meters, so there are 40 meters of ship left where your friend Lucy could be. Following our calculation from before, the 5 meter rear is now $12.5\\%$ of the possible space.[^9]

[^9]: $5/40\cdot100=12.5$

Therefore the probability to meet at least one Lucy at the rear if one of them is certainly _not_ in the middle is $p(I\vert\neg\mathrm{middle})=0.1+0.125=0.225$, i.e. $22.5\\%$. The funny looking elbow thingy $\neg$ means _not_ in probability lingo, though, especially in programming, ‘!’ is also often used.

{% include figures/not_likelihood.html %}

If we know the probability to meet the other Lucy at the rear if your friend _is_ in the restaurant, and also if she’s _not_, we can now also say what the total probability of meeting any of the two at the rear actually is. We only need to add up the _areas_ from our probability square! Just like any rectangular area, they are defined by a width and a height.

{% include figures/total_prob.html %}

The first area (yellow) is defined by the prior (width) and the likelihood (height) while the second area (red) is defined by the “not prior” and “not likelihood”. Mathematically, we can write this as follows:

$$
\begin{aligned}
p(I)&=\sum_{e\in(\mathrm{middle},\neg\mathrm{middle})}p(I,e)\\
    &=\sum_{e\in(\mathrm{middle},\neg\mathrm{middle})}p(I\vert e)p(e)\\
    &=p(I\vert\mathrm{middle})p(\mathrm{middle})+p(I\vert\neg\mathrm{middle})p(\neg\mathrm{middle})\\
    &=0.1\cdot0.72+0.225\cdot0.28=0.135
\end{aligned}
$$

This quantity, $p(I)=\color{yellow}{\rule{1.44cm}{0.2cm}}\color{black}\ +\ \color{red}{\rule{0.56cm}{0.45cm}}$, is _also_ called _evidence_, even though it is the probability of _seeing_ the evidence. The process of removing one of the random variables, in this case the exact position $e$, by enumerating all of its possible values, is called _marginalization_.

**The Posterior:** Now we’re finally in a position to answer the questions from the beginning: The probability that Lucy is in the restaurant _given_ the new information. After all, we would rather spend a few hours calculating than potentially wasting 10 minutes by going to the rear to check it out. This final quantity is called the _posterior_ and we write it as $p(\mathrm{middle}\vert I)$.

Can you guess how to compute this? We’ve already seen that one can write a joint distribution as the product of its marginal and conditional distribution such that $p(\mathrm{middle},I)=p(\mathrm{middle}\vert I)\cdot p(I)$. Solving this for the desired posterior we get:

$$
p(\mathrm{middle}\vert I)=\frac{p(\mathrm{middle},I)}{p(I)}
$$

Now we can use the same trick again, but this time we factorize $p(\mathrm{middle},I)$ into $p(I\vert\mathrm{middle})\cdot p(\mathrm{middle})$. Putting it back into the equation above, we have found a way of expressing the posterior by the _likelihood_, _prior_ and _evidence_ which are all known quantities:

$$
\begin{aligned}\mathrm{posterior}&=\frac{\mathrm{likelihood}\cdot\mathrm{prior}}{\mathrm{evidence}}=\frac{\color{yellow}{\rule{1.44cm}{0.2cm}}}{\color{yellow}{\rule{1.44cm}{0.2cm}}\color{black}\ +\ \color{red}{\rule{0.56cm}{0.45cm}}}\\p(\mathrm{middle}\vert I)&=\frac{p(I\vert\mathrm{middle})\cdot p(\mathrm{middle})}{p(I)}\\&=\frac{0.1\cdot0.72}{0.135}=0.53\end{aligned}
$$

And there you have it, **Bayes’ theorem**! The result might be surprising and, in my opinion, couldn’t have been guessed without doing the calculations. Note that, while $53\\%$ is a lot less than the $72\\%$ you assigned in the beginning, it is still a lot more than the $20\\%$ probability of finding one of the Lucys at the rear if you didn’t know anything about them and therefore didn’t (and shouldn’t) have any preferences. You have a new belief about Lucys whereabouts now, but what should you do if another passenger tells you that she has talked to a Lucy moments ago at the middle of the ship? You simply make your current belief, the posterior probability we have calculated above, the new prior and begin again!

What’s so great about this simple formula is that it tells you how to update your belief in light of new information. This is a very general problem which you also face in your own life all the time and finding the appropriate response based on the strength of your conviction and the evidence is a non-trivial task.

Play with the numbers if you like, to get an intuition about how things change:

<div>
    <input type="text" id="likelihood" placeholder="likelihood"/>
    <input type="text" id="prior" placeholder="prior"/>
    <input type="text" id="evidence" placeholder="evidence"/>
    <button onclick="compare()">Calculate</button>
</div>
<p>Posterior: <a id="posterior">53</a>%</p>

<script>
    function compare(){
        var likelihood = parseFloat(document.getElementById('likelihood').value);
        var prior = parseFloat(document.getElementById('prior').value);
        var evidence = parseFloat(document.getElementById('evidence').value);
        var posterior = (likelihood * prior) / evidence * 100;
        document.getElementById('posterior').innerHTML = posterior.toFixed(2);
	};
</script>
Finally, the YouTuber 3Blue1Brown made a [fantastic video](https://www.youtube.com/watch?v=HZGCoVF3YvM) about Bayes’ theorem in a visual manner[^10], so I highly recommend checking it out if you’re still a bit confused.

[^10]: From where I’ve shamelessly stolen the idea of the probability square.

The code for the visualizations is available [here](https://github.com/hummat/hummat.github.io/blob/master/notebooks/looking-for-lucy.ipynb) and you can play with it here: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/hummat/hummat.github.io/master?filepath=%2Fnotebooks%2Flooking-for-lucy.ipynb)

---
