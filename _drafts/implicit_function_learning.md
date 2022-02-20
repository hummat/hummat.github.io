---
layout: post
title: Implicit Function Learning
abstract: "There are many ways to represent 3D shapes, from the ubiquitous triangle mesh to voxel grids and point clouds. And then there is the one to rule them all: The implicit function."
tags: [3D, deep learning, reconstruction, completion]
category: learning
mathjax: True
jquery: True
plotly: True
update: 0000-01-01
time: 0
words: 0
---

# {{ page.title }}

## What is an implicit function?

You are probably familiar with the concept of functions from mathematics and programming. Both have in common that you provide one or several input(s) $x$ and obtain one or several output(s) $y$ as in $f(x)=y$. Because we want to understand functions in the context of 3D shape representation, consider the function describing the surface of a sphere centered at the origin
$$x=\cos(\alpha)\sin(\beta)r\\ y=\sin(\alpha)\sin(\beta)r\\ z=\cos(\beta)r$$
where $\alpha$ and $\beta$ are the azimuth and polar angle and $r$ is the radius of the sphere. For a sphere with radius one ($r=1$) and angles $\alpha=30^{\circ},\beta=55^{\circ}$ we obtain $(x,y,z)=(0.71,0.41,0.57)$.

TODO: Visualize
<img src="/images/2dgauss.png">
<div data-include="/figures/cup.html"></div>

This is an _explicit_ or _parametric_ way of defining the function: Given the sphere radius, we can directly compute all point coordinates on its surface for arbitrary spherical coordinates $\alpha,\beta$ as $f(\alpha,\beta,r)=(x,y,z)$.

Here is the same function defined _implicitly_:
$$f(x,y,z)=x^2+y^2+z^2=r^2$$. This equation provides the squared distance to the spheres surface from any point $(x,y,z)$ which is easier to see if we rewrite it as $x^2+y^2+z^2-r^2=0$ which is the standard form of implicit functions. All points for which the equation returns zero lie on the surface of the sphere. Further, we know that all points for which $f(x,y,z)<0$ must lie _inside_ while those for which $f(x,y,z)>0$ must lie on the _outside_ of the sphere. Therefore, our function returns a _signed distance_ and computing it for all points inside the volume we obtain a _signed distance field_ or _SDF_. In doing so, we have implicitly defined the spheres surface as the so called _"zero-level set"_, a concept you might be familiar with in the context of elevation lines on topological maps.

TODO: Visualize

This is all good and well you might think, but why bother?

## Why are they useful?

Remember the simple but powerful equation describing _all_ points on the surface of a sphere with arbitrary radius introduced in the beginning. Now try to find a similar equation for the following shape:

TODO: Visualize Stanford bunny

Yeah, me neither. What we still can do rather efficiently though is to compute, for every point, whether it lies within or outside of the bunny. To do so, we select a second point far off in the distance that's surely not within the shape, and connect it with our point of interest[^1]. We then perform a triangle intersection test (as we do have a triangle mesh of our bunny). For every triangle, we first extend it indefinitely to form a plane and compute the intersection of our line (or ray) with it (provided line and plane aren't perpendicular which we can check by computing the dot product of the planes normal vector and the line). If plane and line intersect, we check if the point of intersection lies to the left of all triangle edges. If so, the triangle and the line intersect[^2]. Now, an even number of intersections means the point of interest lies outside of the shape (we have entered and exited), while an uneven number implies we are on the inside.

TODO: Visualize boundary test

Repeating this process many times provides a separation of the space into exterior, interior and _in between_, where _in between_ describes the surface of the shape _implicitly_.

TODO: Visualize exterior and interior

Instead of only computing whether the point of interest is inside our outside of the shape we can also compute its distance where a positive distance means outside and a negative inside as in the sphere example from the beginning. This is a little more involved for general shapes and often not necessary as we will see later. 

[^1]: This is sometimes called "shooting a ray from an arbitrary viewpoint". 
[^2]: Check out [this site](https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution) if you are curious and want more details.

## How can they be learned?

This in between area can be understood as a decision boundary in a binary classification problem with the two classes _outside_ and _inside_.

## Comparing binary occupancies to SDFs

## Applications

## Code & References

| [Code](/url/to/notebook.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](/url/to/binder/notebook.ipynb) |
| :----------------------------: | :-----------------------------------------------------------------------------: |
|                                |                                                                                 |
|              [1]               |                   [Title of a Reference Paper](/url/to/paper)                   |
