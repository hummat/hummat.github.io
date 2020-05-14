---
layout: post
title: Phd? Yes! But what about?
abstract: An ongoing list of potential PhD topics. I thought it might be helpful to put this list here to motivate me but also to be able to easily share it and potentially to get some input from elsewhere. So feel free to comment if you have any great ideas!
tags: [phd, deep learning, machine learning, robotics]
category: brainstorming
update: 2020-05-14
---

# A robotics & machine learning PhD topics list

Here I would like to collect potential topics for my PhD which are of great interest to me. This is mostly written for myself, to order and collect my thoughts. But maybe you are interested in similar topics in which case let’s explore this exciting landscape together. Whatever ends up in this list will have a really high chance to be machine learning related though. Definitely feel free to drop me a comment if you have other cool ideas.

## 1. Field: Bayesian Deep Learning

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

## 2. Field: Combining model and data driven methods

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
