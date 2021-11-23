---
layout: post
title: Implicit Function Learning
abstract: There are many ways to represent 3D shapes, from the ubiquitous triangle mesh to voxel grids and point clouds. And then there is the one to rule them all{:} The implicit function.
tags: [3D, deep learning, reconstruction, completion]
category: learning
mathjax: True
update: 0000-01-01
time: 0
words: 0
---

# {{ page.title }}

## What is an implicit function?

You are probably familiar with the concept of functions from mathematics and programming. Both have in common that you provide one or several input(s) $x$ and obtain one or several output(s) $y$ as in $f(x)=y$. Because we want to understand functions in the context of 3D shape representation, consider the function describing the surface of a sphere centered at the origin
$$x=\cos(\alpha)\sin(\beta)r\\ y=\sin(\alpha)\sin(\beta)r\\ z=\cos(\beta)r$$
where $\alpha$ and $\beta$ are the azimuth and polar angle and $r$ is the radius of the sphere. This is an _explicit_ or _parametric_ way of defining the function: Given the sphere radius, we can directly compute all point coordinates on its surface for arbitrary spherical coordinates $\alpha,\beta$ as $f(\alpha,\beta,r)=(x,y,z)$.
Here is the same function defined _implicitly_:
$$f(x,y,z)=x^2+y^2+z^2=r^2$$.
This equation provides the squared distance to the spheres surface from any point $(x,y,z)$ which is easier to see if we rewrite it as $x^2+y^2+z^2-r^2=0$ which is the standard form of implicit functions. All points for which the equation returns zero lie on the surface of the sphere. Further, we know that all points for which $f(x,y,z)<0$ must lie _inside_ while those for which $f(x,y,z)>0$ must lie on the _outside_ of the sphere. Therefore, our function returns a _signed distance_ and computing it for all points inside the volume we obtain a _signed distance filed_ or _SDF_. In doing so, we have implicitly defined the spheres surface as the so called _"zero-level set"_, a concept you might familiar with in the context of elevation lines on topological maps.

This is all good and well you might think, but why bother?

## Why are they useful?

## How can they be learned?

## Comparing binary occupancies to SDFs

## Applications

## Code & References

| [Code](/url/to/notebook.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](/url/to/binder/notebook.ipynb) |
| :----------------------------: | :-----------------------------------------------------------------------------: |
|                                |                                                                                 |
|              [1]               |                   [Title of a Reference Paper](/url/to/paper)                   |
