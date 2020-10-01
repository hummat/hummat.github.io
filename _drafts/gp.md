---
layout: post
title: Getting Gaussian Processes
abstract: When you work on Bayesian machine learning, you'll inevitably stumble upon Gaussian Processes. You probably won't understand them and then dismiss them as not that important after all. An then they'll appear again. And again. And again. Okay, enough! Le'ts have another look then...
tags: [Gaussian Process, Machine Learning, Deep Learning, Neural Networks]
category: learning
mathjax: True
update: 0000-01-01
time: 0
words: 0
---

# {{ page.title }}

* A Gaussian process is a generalization of the Gaussian probability distribution. Whereas a probability distribution describes random variables which are scalars or vectors (for multivariate distributions), a stochastic process governs the properties of functions.
* One of the main attractions of the Gaussian process framework is precisely that it unites a sophisticated and consistent view with computational tractability.
* Many models that are commonly employed in both machine learning and statistics are in fact special cases of, or restricted kinds of Gaussian processes.
* Regularization corresponds to a restriction bias, while the use of priors corresponds to a preference bias.
* The problem of learning in Gaussian processes is exactly the problem of finding suitable properties for the covariance function.
* Weight space view: Probabilistic linear regression with basis functions
* Length scale: Roughly the distance you have to move in input space before the function value can change significantly
* Function space view: 
* Kernel trick