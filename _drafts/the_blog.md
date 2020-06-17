---
layout: post
title: Making a Blog
abstract: Here I'll explain how this blog came to be, why I chose GitHub Pages for it and some insights I gained while doing so.
tags: [blog, github, git, jekyll]
category: resource
mathjax: True
update: 2020-06-17
time: 1
words: 832
---

# {{ page.title }}

If you’ve read my [first post](https://hummat.github.io/thought/2020/05/11/hello-world.html), you’ll know that I have been toying with the idea to start my own blog for quite some time. Now that I’ve finally done it, I’d like to share how it went so far and what I’ve learned.

## Choosing a platform

Sites like [Medium](https://medium.com/) and [towards data science](https://towardsdatascience.com/) have helped me a good deal in understanding difficult to grasp subjects, because the authors often rely on a simplified language and lot’s of visualizations. There further is discussion section at the end of each article (a vital component for a blog discussing technical topics, as I’ve mentioned [here](https://hummat.github.io/thought/2020/05/28/writing-good-articles.html)) to interact with your audience and to answer questions, explain parts in more detail and iron out mistakes in the article.

The downside of these and similar sites is, that you don’t have a lot of freedom to design your post or add functionality (to which we come in the next section) and it’s yet another account you need to create. Finally, you’ll probably not learn as much compared to setting it up yourself.

Once you’ve decided to take things into your own hand there are of course behemoths like [WordPress](https://wordpress.com/) or [Squarespace](https://www.squarespace.com/) chivalrously extending their hands to offer you creative freedom without friction, but it will cost you. Need a domain? Get on board with our premium plan! Need this feature? Buy this plugin! Want to do this very specific thing? No, we don’t have it, but you can program it yourself…

To be clear, I’m not generally against paying for digital products, even if there are free alternatives, because as you know, if it’s free you are the product (or rather, the resource). But what if there was a free _and_ open source alternative, you can program a little bit and want to learn something new? Let me introduce you to:

## GitHub Pages

[**GitHub** Pages](https://pages.github.com/) let’s you transform any GitHub repository into a website. In fact, this can be done with a click of a button if you’ve already written a great readme. But you can also setup a special repository which is not connected to any project but rather to your GitHub account and is therefore ideal to becoming your mouthpiece.

As you might have noticed, this section (and also the upcoming ones) expects you to know what [git](https://git-scm.com/) and [GitHub](https://github.com/) are. If you’ve never used them, don’t worry, it’s not super complicated to get started, but it’s squarely outside the scope of this article.

### Who is it for?

In fact, let’s quickly determine who the target audience for this approach might be:

1. You want to blog (like a hacker).
2. You don’t want to spend money on it but rather time (and potentially learn something)
3. You want to have (almost) full control over every aspect of the design and functionality.
4. You’ve already worked with git and GitHub.
5. You know how to program (at least a little bit).
6. You don’t hate [Markdown](https://guides.github.com/features/mastering-markdown/)

I’d say 1 to 3 are the most important, because if you want to do something completely different than blogging, say a shopping site, GitHub pages is certainly the wrong thing. And if you don’t mind spending money and are happy with (potentially rigid) templates and plugins, there are also better options out there.

4 and 5 aren’t actually that important, because you can set everything up without any git, GitHub or programming knowledge and once you have, you can gradually start understanding everything and playing with the components until you grasp the basic concepts. It will take a lot more time though.

### How to set it put?

There are a ton of good tutorials out there to setup a basic GitHub page, but I’d like to highlight two of them:

1. [fast.ai](https://www.fast.ai/2020/01/20/blog_overview/): You might know them from their free deep learning course, but they have also published this blog post, that teaches you how to setup your GitHub page without programming or touching the command line at all.
2. [Jekyll Now](https://www.smashingmagazine.com/2014/08/build-blog-jekyll-github-pages/): This is the tutorial I’ve been using for setting up my GitHub page.

## Adding functionality

This is the heart of this article. As I already had [clear ideas](https://hummat.github.io/thought/2020/05/28/writing-good-articles.html) what I expected from my blog, there were quite a few additions I had to make to the basic setup to get everything up and running. All of these are documented _somewhere_ on the Internet, but I think it is a good idea to put them here in one place and to mention some pitfalls when combining everything.

### 1. Working offline

### 2. The discussion section

### 3. Math support

### 4. Interactive visualizations