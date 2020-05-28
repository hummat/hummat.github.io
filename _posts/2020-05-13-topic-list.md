---
layout: post
title: Phd? Yes! But what about?
abstract: An ongoing list of potential PhD topics. I thought it might be helpful to put this list here to motivate me but also to be able to easily share it and potentially to get some input from elsewhere. So feel free to comment if you have any great ideas!
tags: [phd, deep learning, machine learning, robotics]
category: brainstorming
mathjax: true
update: 2020-05-27
---

# A robotics & machine learning PhD topics list

This is an ongoing list of potential topics for my PhD. This is mostly written for myself, to order and collect my thoughts. But maybe you are interested in similar topics in which case we can explore this exciting landscape together. Whatever ends up in this list will have a really high chance to be machine learning related though. Definitely feel free to drop me a comment if you have other cool ideas.

## 1. Bayesian Deep Learning

This is of course a super large field already, but also an extremely interesting one. I’ve been working on a technique called _Laplace Approximation_ during my Masters’ thesis where I applied it to deep neural networks to transform them post hoc into Bayesian neural networks.

There are a myriad of other methods to achieve this, like _Monte Carlo dropout_, ensemble learning and variational approaches.

### Pros:

* Extremely young field, even younger than deep learning, so there is still a lot to discover.
  * Related: Booming field so potentially high impact. Though that’s usually of little interest to me.
* Very important for deep learning to make algorithms deployed in the real world more reliable and therefore less dangerous. I really like the idea to work on AI safety from this angle.
* Principled yet applied. Bayesian probability theory is mature and technical but can be nicely applied to deep learning so it should be a great mix of theory and application.

### Cons:

* Too large. You can’t just study “Bayesian Deep Learning”. I would have to focus on a much smaller subfield, but I’ve no idea yet what that could be.

### Topics:

* Reliable and fast uncertainty estimation for robotics

* Uncertainty estimation for robust robotic perception

## 2. Combining model and data driven methods

I’ve studied robotics for my Masters’ degree and consequently now work with robots. A robot is a very complex system, combining mechanical, electrical and software engineering. A lot of this is well studied and deterministic, following established physical rules. For example, knowing all the parameters of a robotic arm like joint friction, motor torque and the initial position, one can easily compute the motion of the arm when applying voltage to the motors, or reversely, the voltage required to perform a desired motion.

Enter machine learning, or even worse, deep learning. Exit determinism. Because deep learning methods are data driven, learning statistical regularities from it without or with minimal human oversight, the results are inherently opaque to us. Worse, the less we know beforehand, the more data we typically need to get our system to do what we want.

The solution: Apply as much prior knowledge about the system as possible in the form of physical models and only learn the remaining unsolved parts, i.e. those for which we don’t have closed form equations. In our example from above this might translate into an algorithm that already knows how to move the arm around by applying certain voltages to certain motors in certain joints an when given the task to, e.g., solve a Rubik’s cube, it only needs to learn the solution to the problem itself, i.e. what to turn when and where, instead of also having to learn how to move.

This is a lot like an adult solving a problem vs a baby. While the baby first needs to learn how to move its extremities and what “solve the Rubik’s cube” even means, the adult can directly proceed to twisting and turning.

### Pros:

* Quite the rage at the moment. As large data sets are expensive to create there is high demand for more efficient methods.
* Especially relevant in robotics due to the close connection of well established “old school” fields such as mechanical and electrical engineering and cutting edge deep learning for perception.
* Elegant, as we don’t reinvent the wheel for each task but instead stand on the shoulders of giants.
* Suits me, as I have some prior knowledge due to a Bachelor’s degree in mechanical engineering.

### Cons:

* Difficult because one needs in depth knowledge of machine learning _and_ physics.
* Also by far too large. I would have to find a concrete use case or much smaller subtask.

### Topics:

* Bayesian motion learning through model priors

## 3. AI Safety

From the paper [Concrete Problems in AI Safety](https://arxiv.org/pdf/1606.06565.pdf): _Rapid progress in machine learning and artificial intelligence (AI) has brought increasing attention to the potential impacts of AI technologies on society. In this paper we discuss one such potential impact: the problem of accidents in machine learning systems, defined as unintended and harmful behavior that may emerge from poor design of real-world AI systems. We present a list of five practical research problems related to accident risk, categorized according to whether the problem originates from having the wrong objective function ("avoiding side effects" and "avoiding reward hacking"), an objective function that is too expensive to evaluate frequently ("scalable supervision"), or undesirable behavior during the learning process ("safe exploration" and "distributional shift"). We review previous work in these areas as well as suggesting research directions with a focus on relevance to cutting-edge AI systems. Finally, we consider the high-level question of how to think most productively about the safety of forward-looking applications of AI._

I think AI safety is an extremely important field of research, particularly now, while we still have some time to get it right _before_ we deploy more and more machine learning solutions into the real world. It is also highly relevant for robotics, especially if those robots are to be used alongside humans. The authors discuss the following topics:

### Topics

- **Safe exploration.** *Can [reinforcement learning](http://karpathy.github.io/2016/05/31/rl/) (RL) agents learn about their environment without executing catastrophic actions?* For example, can an RL agent learn to navigate an environment without ever falling off a ledge?
- **Robustness to distributional shift.** *Can machine learning systems be robust to changes in the data distribution, or at least fail gracefully?* For example, can we build [image classifiers](https://www.tensorflow.org/versions/r0.9/tutorials/deep_cnn/index.html) that indicate appropriate uncertainty when shown new kinds of images, instead of confidently trying to use its [potentially inapplicable](http://arxiv.org/abs/1412.6572) learned model? -> Bayesian Deep Learning!
- **Avoiding negative side effects.** *Can we transform an RL agent’s [reward function](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node9.html) to avoid undesired effects on the environment?* For example, can we build a robot that will move an object while avoiding knocking anything over or breaking anything, without manually programming a separate penalty for each possible bad behavior?
- **Avoiding “reward hacking” and “[wireheading](http://www.agroparistech.fr/mmip/maths/laurent_orseau/papers/ring-orseau-AGI-2011-delusion.pdf)”.** *Can we prevent agents from “gaming” their reward functions, such as by distorting their observations?* For example, can we train an RL agent to minimize the number of dirty surfaces in a building, without causing it to avoid looking for dirty surfaces or to create new dirty surfaces to clean up?
- **Scalable oversight.** *Can RL agents efficiently achieve goals for which feedback is very expensive?* For example, can we build an agent that tries to clean a room in the way the user would be happiest with, even though feedback from the user is very rare and we have to use cheap approximations (like the presence of visible dirt) during training? The divergence between cheap approximations and what we actually care about is an important source of accident risk.

Reinforcement learning, while super interesting, is not really my cup of tea so the second topic might be most interesting to me, especially as it plays nicely with Bayesian approaches. The last topic can also be applied to non-RL scenarios where it is usually called _active learning_ but with limited data.

## 4. 3D Deep Learning

The main difference (and problem) compared to 2D data like images is the permutation invariance of individual 3D data like point clouds. So given $N$ points, there are $N!$ ways to feed those to the learning algorithm while all should result in the same classification or segmentation result.

Several ideas have been proposed to deal with this problem from taking multiple _views_ of the object and using a standard CNN on those to transforming the point cloud into a graph structure or working directly on the point cloud but making use of local neighborhoods.

### Pros

* Very young field so probably lots of opportunities to improve upon existing approaches.

### Cons

* It is not entirely clear to me if 3D data will actually be important in the near future because it might be that RGB based approaches (i.e. cameras) are enough for doing everything we care about (“humans don’t use depth sensors”).

