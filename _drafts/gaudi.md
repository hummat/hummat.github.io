---
layout: post
title: Gaudí in three dimensions
abstract: A first look at turning smartphone captures from Barcelona into interactive Gaussian splats.
category: scenes
tags: [3dgs, gaussian-splatting, barcelona]
time: 3
words: 250
spark_viewer: true
# Switch to https://assets.hummat.com/data/splats before publishing.
splat_base: /_assets/data/splats
---

Most travel photos flatten a place into a collection of rectangles. For this first scene post, I wanted to keep a
little of the third dimension. These captures come from smartphone videos of Gaudí-related scenes in Barcelona,
reconstructed as interactive Gaussian splats.

The format is still a test: a short paragraph, a scene, and enough room to rotate it yourself. Click **Load
interactive view**, then drag to orbit and scroll to zoom. Of course, the results are not equally good, but that is
part of the point. A capture can look perfectly reasonable while recording it and turn out rather differently after
reconstruction.

{% assign splats = "gaudi_ceiling,gaudi_chair_1,gaudi_chair_1_b,gaudi_chair_2,gaudi_chair_3,gaudi_chair_4,gaudi_chimneys,gaudi_fountain,gaudi_lizard,gaudi_gramophone,gaudi_office,gaudi_phone,gaudi_plant,gaudi_portemonteau,gaudi_wall,gaudi_window" | split: "," %}
{% for splat in splats %}
{% assign scene_url = page.splat_base | append: "/" | append: splat | append: ".sog" %}
{% assign scene_name = splat | replace: "_", " " %}
{% comment %} Toggle wiring (unfiltered_url=...) returns once the manual cleanup pass produces the final PLYs; the meaningful comparison is then auto-crop PLY vs manually cleaned PLY, both from the WEB250k checkpoints. {% endcomment %}
{% include spark-viewer.html url=scene_url alt=scene_name caption=scene_name %}
{% endfor %}

This is the first pass at the scenes format. The captures are the content; the writing only needs to get out of the
way and tell you what you are looking at.
