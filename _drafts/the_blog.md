---
layout: post
title: Blog in the making
abstract: Here I'll explain how this blog came to be, why I chose GitHub Pages for it and some insights I gained while doing so.
tags: [blog, github, git, jekyll]
category: resource
mathjax: True
update: 2020-06-17
time: 1
words: 832
---

# {{ page.title }}

If you’ve read my [first post](https://hummat.github.io/thought/2020/05/11/hello-world.html), you’ll know that I have been toying with the idea to start my own blog for quite some time. Now that I’ve finally done it, I’d like to share how it went so far and what I’ve learned till now (disclaimer: a lot!). If you already have your own GitHub page and only want some information on how to integrate feature X you’ve seen on this blog, you can skip ahead to [Adding functionality](#adding-functionality) immediately.

## Choosing a platform

Sites like [Medium](https://medium.com/) and [towards data science](https://towardsdatascience.com/) have helped me a good deal in understanding difficult to grasp subjects, because the authors often rely on a simplified language and lot’s of visualizations. Further, there is a discussion section at the end of each article (a vital component for a blog discussing technical topics, as I’ve suggested [here](https://hummat.github.io/thought/2020/05/28/writing-good-articles.html)) to interact with your audience and to answer questions, explain parts in more detail and iron out mistakes in the article.

The downside of these and similar sites is, that you don’t have as much freedom to design your post or add functionality (to which we come in the final section), you are dependent on the site operator and its policies (like financing it through adds) and it’s yet another account you need to create. Finally, you’ll probably not learn as much compared to setting it up yourself.

Once you’ve decided to take things into your own hand there are of course behemoths like [WordPress](https://wordpress.com/) or [Squarespace](https://www.squarespace.com/) chivalrously extending their hands to offer you creative freedom without friction, but it will cost you. Need a domain? Get on board with our premium plan! Need this feature? Buy this plugin! Want to do this very specific thing? No, we don’t have it, but you can program it yourself…

To be clear, I’m not generally against paying for digital products, even if there are free alternatives, because as you know, if it’s free you are the product (or rather, the resource). But what if there was a free _and_ open source alternative, you can program a little bit and want to learn something new? Let me introduce you to:

## GitHub Pages

[**GitHub** Pages](https://pages.github.com/) let’s you transform any GitHub repository into a website. In fact, this can be done with a click of a button if you’ve already written a great readme. But you can also setup a special repository which is not connected to any project but rather to your GitHub account and is therefore ideal to becoming your mouthpiece.

As you might have noticed, this section (and also the upcoming ones) expects you to know what [git](https://git-scm.com/) and [GitHub](https://github.com/) are. If you’ve never used them, don’t worry, it’s not super complicated to get started, but it’s squarely outside the scope of this article to introduce them.

### Why is it great?

Here are some advantages of this approach:

1. It’s foss! Free and open source.
2. It feels familiar. Chances are, if you’ve ever programmed something, especially collaboratively, you’ve used git and you probably already have a GitHub account. Using this familiar technology feels natural and you don’t need yet another account.
3. You get the usual benefits of git but for web development: top notch version control and easy collaboration.
4. You can write all the content in markdown! I love markdown, especially for quickly putting together and structuring a thought, so ideal for blogging. You’re markdown content is converted to HTML automatically, so you don’t need to write _a single_ line of HTML, CSS or JavaScript (but you can)!
5. No database: You’re sites repository functions as hub for all the content displayed on your page so it’s extremely fast (short loading times) and there is no (My)SQL involved, which also means no dependency on external services.
6. Secure: No content management system (i.e. the online tools found in products like WordPress to edit your page) and no database or use of PHP (which means you produce _static_ sites, which still can be interactive though).

### Who is it for?

Let’s quickly determine who the target audience for this approach might be:

1. You want to blog (like a hacker).
2. You don’t want to spend money on it but rather time (and potentially learn something)
3. You want to have (almost) full control over every aspect of the design and functionality.
4. You’ve already worked with git and GitHub.
5. You know how to program (at least a little bit).
6. You don’t hate [Markdown](https://guides.github.com/features/mastering-markdown/)

I’d say 1 to 3 are the most important, because if you want to do something completely different than blogging, say a shopping site, GitHub pages is certainly the wrong thing. And if you don’t mind spending money and are happy with (potentially rigid) templates and plugins, there are also better options out there.

4 and 5 aren’t actually that important, because you can set everything up without any git, GitHub or programming knowledge and once you have, you can gradually start understanding everything and playing with the components until you grasp the basic concepts. It will take a lot more time though.

### How to set it up?

There are a ton of good tutorials out there to setup a basic GitHub page, but I’d like to highlight two of them:

1. [fast.ai](https://www.fast.ai/2020/01/20/blog_overview/): You might know them from their free deep learning course, but they have also published this blog post, that teaches you how to setup your GitHub page without programming or touching the command line at all.
2. [Jekyll Now](https://www.smashingmagazine.com/2014/08/build-blog-jekyll-github-pages/): This is the tutorial I’ve been using for setting up my GitHub page.

No matter which way you choose, I highly recommend using a template. What? Didn’t you just say that you don’t like templates? Well, there are at least two types of templates. There are those which I’d call _closed source_ or cryptic, so you either can’t or are highly discouraged to change anything major. And then there are those which are actually in the spirit of a true template: a starting point from where you can go _everywhere_ (not just a few timid steps in a couple of predefined directions). I’m talking about the latter here.

The reason templates are so great is, that instead of reinventing the wheel, which is very time consuming and difficult, you merely need to tinker with it and see what happens. After some back and forth you start to get an understanding which part does what and how they work together. A great template is [Minima](https://github.com/jekyll/minima) which already features a lot of functionality and is ready for offline work, to which I’ll come next, among other things.

## Adding functionality

This is the heart of this article. As I already had [clear ideas](https://hummat.github.io/thought/2020/05/28/writing-good-articles.html) what I expected from my blog, there were quite a few additions I had to make to the basic setup to get everything up and running. All of these are documented _somewhere_ on the Internet, but I think it is a good idea to put them here in one place  and how to make them play together as there are some pitfalls when combining everything.

### 1. Working offline

Being able to work offline is extremely useful for a variety of reasons. First of all, you can work wherever and whenever you like. Secondly, you don’t rely on any service being operational, because even if you have an Internet connection, your host (i.e. GitHub) might be down or there are other connection problems. It is also faster, especially when working with GitHub pages, because once pushed, your changes aren’t visible immediately. And as you’re basically programming your blog, you will make mistakes and you will want to check what a new feature looks like regularly, so this can become quite annoying. Fortunately, it is quite easy to set this up, because GitHub pages works with Jekyll behind the scenes to convert your markdown into HTML and you can run Jekyll locally.

> **What’s Jekyll?** Jekyll is a static site generator with built-in support for GitHub Pages and a simplified build process. Jekyll takes Markdown and HTML files and creates a complete static website based on your choice of layouts. Jekyll supports Markdown and Liquid, a templating language that loads dynamic content on your site

I’ll assume a couple of things before moving forward:

1. You’re on Linux.
2. You have git installed and know how to use it.
3. You have a GitHub account.
4. You know [how to setup a basic GitHub page](#how-to-set-it-up). 
5. You’ve cloned or forked a template like [Minima](https://github.com/jekyll/minima). You can also have a look at the GitHub repository for [this blog](https://github.com/hummat/hummat.github.io), which, as mentioned previously, is a fork of the [Jekyll Now](https://www.smashingmagazine.com/2014/08/build-blog-jekyll-github-pages/) theme.

Now, follow the official Jekyll Ubuntu or other Linux distros [installation guide](#https://jekyllrb.com/docs/installation/). If you haven’t cloned or forked a template like mentioned in step 5 above, you can also clone your basic GitHub page repository, change into its directory and run `bundle exec jekyll new .` (note the dot in the end). Afterwards, or if you have used a template, run `bundle exec jekyll serve`. If you get errors, run `bundle update` and try again.

There are a couple of interesting files in your GitHub pages repository. For now, the most important is the [Gem]

You should now be able to open your GitHub page in your browser by navigating to `https:\\127.0.0.1:4000`.

 

### 2. The discussion section

### 3. Math support

### 4. Interactive visualizations

### 5. Tips & Tricks

* Combining HTML and Markdown
* Cross-references
* Selective printing