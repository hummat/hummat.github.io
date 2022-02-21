---
layout: post
title: Beautiful Blogging
abstract: How does this blog work? What are all these tagged things? Why GitHub pages? This article answers all of these questions and more!
tags: [blog, github, git, jekyll, plotly, mathjax, binder, disqus]
category: resource
jquery: true
plotly: true
mathjax: True
slideshow: True
thumbnail: /images/github.png
update: 2021-06-24
time: 12-16
words: 3221-4202
---

# {{ page.title }}

If you’ve read my [first article](https://hummat.github.io/thought/2020/05/11/hello-world.html), you’ll know that I have been toying with the idea to start my own blog for quite some time. Now that I’ve finally done it, I’d like to share how it went so far and what I’ve learned till now (disclaimer: a lot!). If you already have your own GitHub page and only want some information on how to integrate feature X you’ve seen on this blog, you can skip ahead to [Adding functionality](#adding-functionality) immediately.

## Choosing a platform

Sites like [Medium](https://medium.com/) have helped me a good deal in understanding difficult to grasp subjects, because the authors often rely on a simplified language and lot’s of visualizations. Further, there is a discussion section at the end of each article (a vital component for a blog discussing technical topics, as I’ve suggested [here](https://hummat.github.io/thought/2020/05/28/writing-good-articles.html)) to interact with your audience and to answer questions, explain parts in more detail and iron out mistakes in the article.

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
4. You can write all the content in markdown! I love markdown, especially for quickly putting together and structuring a thought, so ideal for blogging. Your markdown content is converted to HTML automatically, so you don’t need to write _a single_ line of HTML, CSS or JavaScript (but you can)!
5. No database: Your sites repository functions as hub for all the content displayed on your page so it’s extremely fast (short loading times) and there is no (My)SQL involved, which also means no dependency on external services.
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

No matter which way you choose, I highly recommend using a template. What? Didn’t he just say that he doesn't like templates? Well, there are at least two types of templates. There are those which I’d call _closed source_ or cryptic, so you either can’t or are highly discouraged to change anything major. And then there are those which are actually in the spirit of a true template: a starting point from where you can go _everywhere_ (not just a few timid steps in a couple of predefined directions). I’m talking about the latter here.

The reason templates are so great is, that instead of reinventing the wheel, which is very time consuming and difficult, you merely need to tinker with it and see what happens. After some back and forth you start to get an understanding which part does what and how they work together. A great template is [Minima](https://github.com/jekyll/minima) which already features a lot of functionality and is ready for offline work, to which I’ll come next, among other things.

## Adding functionality

This is the heart of this article. As I already had [clear ideas](https://hummat.github.io/thought/2020/05/28/writing-good-articles.html) what I expected from my blog, there were quite a few additions I had to make to the basic setup to get everything up and running. All of these are documented _somewhere_ on the Internet, but I think it is a good idea to put them here in one place and how to make them play together as there are some pitfalls when combining everything.

### 1. Working offline

Being able to work offline is extremely useful for a variety of reasons. First of all, you can work wherever and whenever you like. Secondly, you don’t rely on any service being operational, because even if you have an Internet connection, your host (i.e. GitHub) might be down or there are other connection problems. It is also faster, especially when working with GitHub pages, because once pushed, your changes aren’t visible immediately. And as you’re basically programming your blog, you will make mistakes and you will want to check what a new feature looks like regularly, so this can become quite annoying. Fortunately, it is quite easy to set this up, because GitHub pages works with Jekyll behind the scenes to convert your markdown into HTML and you can run Jekyll locally.

> **What’s Jekyll?** Jekyll is a static site generator with built-in support for GitHub Pages and a simplified build process. Jekyll takes Markdown and HTML files and creates a complete static website based on your choice of layouts. Jekyll supports Markdown and Liquid, a templating language that loads dynamic content on your site

#### Prerequisites

I’ll assume a couple of things before moving forward:

1. You’re on Linux.
2. You have git installed and know how to use it.
3. You have a GitHub account.
4. You know [how to setup a basic GitHub page](#how-to-set-it-up).
5. You’ve cloned or forked a template like [Minima](https://github.com/jekyll/minima). You can also have a look at the GitHub repository for [this blog](https://github.com/hummat/hummat.github.io), which, as mentioned previously, is a fork of the [Jekyll Now](https://www.smashingmagazine.com/2014/08/build-blog-jekyll-github-pages/) theme.

#### Setup

Now, follow the official Jekyll Ubuntu or other Linux distros [installation guide](https://jekyllrb.com/docs/installation/). If you haven’t cloned or forked a template like mentioned in step 5 above, you can also clone your basic GitHub page repository, change into its directory and run `bundle exec jekyll new .` (note the dot in the end).

There are a couple of interesting files in your GitHub page repository/directory. For now, the most important is the [Gemfile](https://github.com/hummat/hummat.github.io/blob/master/Gemfile). Here you need to change `gem “jekyll”` into `gem "github-pages", "~> 206", group: :jekyll_plugins`. You can also go ahead and add some plugins we will use later:

```javascript
group :jekyll_plugins do
  gem "jekyll", "~> 3.8.7"
  gem "jekyll-sitemap", "~> 1.4.0"
  gem "jekyll-feed", "~> 0.13"
  gem "jekyll-seo-tag", "~> 2.6.1"
end
```

The number behind each plugin is its version, which should match the GitHub pages [dependency version](https://pages.github.com/versions/) to make sure that your offline site looks and works identical to the one processed by GitHub once you go online. Now, head into the [`_config.yml`](https://github.com/hummat/hummat.github.io/blob/master/_config.yml) file and add/change `markdown: kramdown`, which is the processor used by GitHub to convert your markdown into HTML.

Afterwards, or if you have used a template, run `bundle exec jekyll serve`. If you get errors, run `bundle update` and try again. You should now be able to open your GitHub page in your browser by navigating to [http://127.0.0.1:4000/](http://127.0.0.1:4000/).

#### Files and directories

Something I found confusing in the beginning is the rather complicated [file and directory structure](https://github.com/hummat/hummat.github.io). Here’s a quick overview of what’s what:

1. [`index.md`](https://github.com/hummat/hummat.github.io/blob/master/index.md): Your landing page. It’s empty except for its layout, which is defined in [`_layouts/home.html`](https://github.com/hummat/hummat.github.io/blob/master/_layouts/home.html).
2. [`about.md`](https://github.com/hummat/hummat.github.io/blob/master/about.md): Jekyll differentiates between _pages_ and _posts_. I’m using pages as self contained units, like the landing page and the _About_ page and posts for, well, posts.
3. [`style.scss`](https://github.com/hummat/hummat.github.io/blob/master/style.scss): Together with the files in [`_sass`](https://github.com/hummat/hummat.github.io/tree/master/_sass), this determines the look of your HMTL elements.
4. [`_posts`](https://github.com/hummat/hummat.github.io/tree/master/_posts): This is where your blog posts go. It’s just a bunch of markdown files, one for each post with a date in front which gets added to the `page.date` variable automatically. I’ll come back to those variables in the [next section](#liquid-templating).
5. [`_drafts`](https://github.com/hummat/hummat.github.io/tree/master/_drafts): Here you can put unfinished posts which you can render locally using `bundle exec jekyll serve --draft` and which don’t show up in the online version of your page.
6. [`_layouts`](https://github.com/hummat/hummat.github.io/tree/master/_layouts): This directory, together with `_includes`, `_sass` and the `style.css` are the heart of each template and define the entire look of your page, apart from the content of your posts. Here you find all the HTML code of your site, so this is where you should look if you want to change the structure of it. There is a [`default.html`](https://github.com/hummat/hummat.github.io/blob/master/_layouts/default.html) defining some properties which are inherited by the other layout files while the [`home.html`](https://github.com/hummat/hummat.github.io/blob/master/_layouts/home.html), [`page.html`](https://github.com/hummat/hummat.github.io/blob/master/_layouts/page.html) and [`post.html`](https://github.com/hummat/hummat.github.io/blob/master/_layouts/post.html) define the layouts of the index page as well as other pages and posts respectively.
7. [`_includes`](https://github.com/hummat/hummat.github.io/tree/master/_includes): Here you’ll find additional functionality which you can include into a layout. The `default.html` layout for example includes the [`mathjax.html`](https://github.com/hummat/hummat.github.io/blob/master/_includes/mathjax.html) for math support while the `post.html` includes [`disqus.html`](https://github.com/hummat/hummat.github.io/blob/master/_includes/disqus.html) for [DISQUS](https://disqus.com/) discussion support. We’ll look at those includes in more detail in the following sections.

### 2. Liquid templating

[Liquid](https://shopify.github.io/liquid/) is a kind of small scripting language, allowing you to do rudimentary dynamic things inside your otherwise static GitHub page. It’s mainly used to store values in variables, which can then be used elsewhere and for simple if-else control flow (or at least I’m using it like this).

Variables can be accessed using {% raw %}{{ variable }}{% endraw %} while logical statements are written like {% raw %}{% if statement %}{% endraw %}. How to define variables? There are a number of predefined variables you can use anywhere on the site or in a post like the `site.title` and `site.description` defined in the [\_config.yml](https://github.com/hummat/hummat.github.io/blob/master/_config.yml) or `site.posts`, a list of all your posts in the `_posts` directory which is used by [`home.html`](https://github.com/hummat/hummat.github.io/blob/master/_layouts/home.html) to iterate over all of them using {% raw %}{% for post in site.posts %}{% endraw %} and to render them subsequently.

A page also has some predefined variables like `page.date`. Here is where seems between Jekyll and Liquid begin to show in my opinion, as Liquid only knows `site` and `page` so the date of a Jekyll post or page are both accessed using `page.date`. That’s confusing! So always use `page` to access a variable, even if you’re writing a post. For example, I like to begin each post with the following markdown: {% raw %}# {{ page.title }}{% endraw %}.

Finally, variables can be defined by yourself, for example in the [YAML Front Matter](https://jekyllrb.com/docs/front-matter/) of each markdown file. You simply put a

```yaml
---
layout: post
title: And now...The Larch!
thumbnail: /images/image.png
---

```

at the front of a post and whatever you define there can then be accessed using e.g. {% raw %}{{ page.thumbnail }}{% endraw %}. As hinted at in the example, I like to define the location of the thumbnail image I use next to a post here and then to include it in [home.html](https://github.com/hummat/hummat.github.io/blob/master/_layouts/home.html) like so:[^post]

[^post]: Note again the `post` vs `page` dilemma.

```html
{% raw %}{% if post.thumbnail %}{% endraw %}
<div class="thumbnail">
  <img src="{% raw %}{{ post.thumbnail }}{% endraw %}" />
</div>
{% raw %}{% endif %}{% endraw %}
```

By the way, I just learned something new while writing this section: If you want Jekyll to render Liquid code like {% raw %}{{ variable }}{% endraw %} or {% raw %}{% if post.thumbnail %}{% endraw %} instead of evaluating it, you need to put it into {% raw %}{% raw %}{% endraw %} and `endraw` fences instead of simple markdown code fences[^endraw].

[^endraw]: The `endraw` should look like the {% raw %}{% raw %}{% endraw %} but I couldn’t get that to work…

### 3. The discussion section

Enabling a [DISQUS](https://disqus.com/) discussion section involves two steps. First you need to make a DISQUS account. Once you have it, go to `Admin` and at the top create a new site. The _Website name_ should be `yourgithubname.github.io` and the _Website URL_ should be `https://yourgithubname.github.io/`. Once you have created your site you’ll get a `Shortname` which is a unique identifier for your site and which you’ll need for the next step.

Open your[`_config.yml`](https://github.com/hummat/hummat.github.io/blob/master/_config.yml) and add/change `disqus: yourshortname`. Mine is `disqus: hummat-github-io` and I guess yours will be similar. You’ll also have to add/change the `url` field. Mine is `url: "https://hummat.github.io"`. In your [`post.html`](https://github.com/hummat/hummat.github.io/blob/master/_layouts/post.html) add {% raw %}<div data-include="disqus.html"></div>{% endraw %}. Make sure you have a [`disqus.html`](https://github.com/hummat/hummat.github.io/blob/master/_includes/disqus.html) similar to mine in your `_includes` directory. It checks whether you have defined your shortname in the `_config.yml` and disables comments if you’ve set `comments` to `false` inside your posts YAML Front Matter, which you can use to disable comments in certain posts if you don’t need them.

That’s it! You now should have a discussion section similar to the one at the end of this post.

### 4. Math support

If you want to write about math related topics like machine learning, you need to be able to display mathematical equations. While you can get away with simply typing 2+2=4, it is not exactly pretty and anything slightly more complex is out of your reach. The de facto standard to render high quality math in the scientific community is $\LaTeX$. Aha, you see where this is going, right?

On the web, Latex, or rather Latex math mode, is supported through the JavaScript library [MathJax](https://www.mathjax.org/). To get it to work inside GitHub pages you need to have an include file like [this one](https://github.com/hummat/hummat.github.io/blob/master/_includes/mathjax.html). There are a few things to note about this file. The first is the Liquid tag {% raw %}{% if page.mathjax %}{% endraw %}, which allows you to only enable MathJax in posts where you need it, similar to the DISQUS approach above[^javascript].

[^javascript]: We do this, because JavaScript libraries are usually slow to load, especially if they are large, so no point in bloating up a post which doesn’t even make use of the functionality.

We then configure how we want MathJax to behave. Most important are the `inlineMath` and `displayMath` lines, which allow you to enable math mode the same way you’d do it inside a Latex file.

Finally, we load the library from a remote source so it’s always up to date and doesn’t bloat our site. After some research I’m now pretty confident that the URL given in my [`mathjax.html`](https://github.com/hummat/hummat.github.io/blob/master/_includes/mathjax.html) is currently the correct one to use. You can now do things like this!

$$
\mathcal{N}(\mu,\sigma^2)=\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}
$$

### 5. Interactive visualizations

Okay, so this really gets me excited! The ability to add interactive visualizations is my newest achievement and it’s so neat! I’ve already written about why I think visualizations are the best thing that can happen to a complicated topic [here](https://hummat.github.io/thought/2020/05/28/writing-good-articles.html), so I won’t repeat myself, but interactive visualizations?! That’s like visualizations on steroids. So, how can it be done?

It all starts with [Plotly’s Python graphing library](https://plotly.com/python). It’s basically Matplotlib for the web, if you know what I mean. You write some visualization code like [this one](https://github.com/hummat/hummat.github.io/blob/master/notebooks/beautiful-blogging.ipynb)—play with it on [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/hummat/hummat.github.io/master?filepath=%2Fnotebooks%2Fbeautiful-blogging.ipynb) if you like (might take a while to start)—and then you save the figure in a special way:

```python
import plotly.io as pio
pio.write_html(fig,
               file='../_includes/figures/figure.html',
               full_html=False,
               # include_mathjax='cdn',
               include_plotlyjs='cdn')
```

It stores the interactive figure inside HTML file which you can then include in your post using {% raw %}<div data-include="figures/figure.html"></div>{% endraw %}. We set `full_html` to `false` because we want the figure to be part of an already existing HTML site and `include_plotlyjs` to _‘cdn’_ which means the Plotly JavaScript library will be fetched from the web (similar to the MathJax library)[^figmj] because it would otherwise bloat each figure to >3MB. Here’s a comparison (click on the arrows to change the image and on the second image to see how interactive it is!):

[^figmj]: Use `include_mathjax` if you need Latex support in figure labels etc.

<div class="slideshow-container">
  <div class="mySlides fade">
    <div class="numbertext">1 / 2</div>
    <img src="/images/2dgauss.png" style="width:100%">
    <div class="text">A normal image. Boring!</div>
  </div>

  <div class="mySlides fade">
    <div class="numbertext">2 / 2</div>
    <div data-include="figures/figure.html"></div>
    <div class="text">An interactive visualization! Wow!</div>
  </div>

<a class="prev" onclick="plusSlides(-1)">&#10094;</a>
<a class="next" onclick="plusSlides(1)">&#10095;</a>

</div>

### 6. Markdown and HTML: A love-hate relationship

Markdown is great for structuring text with ease (and that’s what it’s made for) but as soon as you’d like to add some advanced functionality like image slideshows or even just colored text, you’ll have to resort to HTML. Because Jekyll converts your markdown to HTML anyways, this often works out of the box. But when it doesn’t work, it’s weird. Part of the problem is the implicit handling of some style properties in markdown by adding or removing whitespace (tabs, spaces, newlines) and another the peculiarities of different markdown-HTML converters.

Take collapsible section for example. They are not natively supported by markdown, but easily to add with a little HTML like this:

```html
<details>
  <summary>Click to expand!</summary>

  ## Heading 1. A numbered 2. list * With some * Sub bullets
</details>
```

Now when I actually type this here, see what happens:

<details>
  <summary>Click to expand!</summary>

## Heading

1. A numbered
2. list
_ With some
_ Sub bullets
</details>

The collapsible is there, but it's content, written in markdown, doesn't get processed into HTML. There are several solutions to this. You can simply write everything in HTML, it works, but that's usually what you were trying to avoid in the first place by using markdown with Jekyll. You can switch the markdown processor for _kramdown_ to [_CommonMarkGhPages_](https://github.com/github/jekyll-commonmark-ghpages) in your `_config.yml`. You have to be careful to add a blank line between the first HTML line and the markdown content and to indent everything, but it works great. Until you want to add some sophisticated HTML, like an [interactive visualization](#interactive-visualization), and you have to find out that it gets partly parsed as markdown to be converted into HTML and breaks.

The solution I’m currently using is the following:

```html
<details markdown="1">
  <summary>Click to expand!</summary>
  ## Heading 1. A numbered 2. list * With some * Sub bullets
</details>
```

This works as expected! (Note the `markdown="1"` addition)

<details markdown="1">
<summary>Click to expand!</summary>
## Heading
1. A numbered
2. list
   * With some
   * Sub bullets
</details>

There is no need to add weird whitespace, doesn't break other code like the interactive visualizations and can also be used to wrap markdown content in other kinds of [HTML containers](https://hummat.github.io/resource/2020/06/08/conferences.html).

### 8. Tips, Tweaks & Tricks

There are a couple of small tweaks and insights I’d like to summarize in this section. This list probably (and hopefully) be growing over time, as I'll add whatever I find out.

- #### Popup footnotes

  Footnotes are great to unclutter your main text, but I hate to have to jump to the end of the page (or, heavens forbid, scroll there) only to scroll back up in search of where I left of once I’ve read the footnote. That’s already slightly improved in the build in way markdown handles footnotes, as there always is a reference _back_ to where you came from. Simply write `[^1]` where you want your footnote to appear and then, somewhere below, `[^1]: your footnote text`.

  But the main advantage of the digital format is, that you can show footnotes in a popup next to the text where they appear whenever you hover or click on them. To enable this functionality, simply add [this file](https://github.com/hummat/hummat.github.io/blob/master/_includes/popup.html) to your `_include` directory[^5] and this line {% raw %}<div data-include="popup.html"></div>{% endraw %} to your e.g. default layout in `_layouts`. Et voilá, popup footnotes![^6]

  [^5]: Adapted from [here](https://github.com/vaetas/hugo-footnotes-popup).
  [^6]: I’m a popup footnote!

  If your footnotes don’t look nice (e.g. the number is too large and not a superscript), add those lines to your `style.scss`:

  ```scss
  sup {
    line-height: 1em;
    font-size: 0.75em;
    position: relative;
    top: -0.5em;
    vertical-align: baseline;
  }
  ```

- #### Cross-references

  To use cross-references in markdown, i.e. jump to a specific section, you can simply use `[something](#the-section-name)` which will then look like [something](#cross-references). If you want to reference to something else than a section, you’ll have to put an HTML anker there first, like so `<a name="this-is-an-anker"></a>` which you can then refer to in the same way as before with `[something](#this-is-an-anker)`.

- #### Selective printing

  In an [earlier post](https://hummat.github.io/book/2020/06/04/deep-work.html) I wanted the reader to be able to print out a specific section (the summary) of the post. One way to achieve this is wrapping everything you _don’t_ want to be printed into `<div class="noprint" markdown="1"></div>`[^7] fences and putting the following into your `style.scss` file:

  ```scss
  @media print {
    .noprint {
      display: none;
    }
  }
  ```

  You can also add additional things here like `footer {display: none;}` if you don’t want your footer to show up in the printout.

  [^7]: Remember the [`markdown="1"` trick](#6-markdown-and-html-a-love-hate-relationship) to force _kramdown_ to process your markdown as such?

- #### Slideshows

  Slideshows are neat, because you can put in tons of visualizations without making your post unbearably long. I’ve basically copy-pasted [this example](https://www.w3schools.com/howto/howto_js_slideshow.asp) into a [`_includes/slideshow.html`](https://github.com/hummat/hummat.github.io/blob/master/_includes/slideshow.html) (JavaScript part)[^8] and a [`_sass/_slideshow.scss`](https://github.com/hummat/hummat.github.io/blob/master/_sass/_slideshow.scss) (CSS part) and enabled it by putting {% raw %}<div data-include="slideshow.html"></div>{% endraw %} into my [`_layouts/default.html`](https://github.com/hummat/hummat.github.io/blob/master/_layouts/default.html) file as well as `@import "slideshow";` at the top of my [`style.scss`](https://github.com/hummat/hummat.github.io/blob/master/style.scss) file.

  [^8]: With the addition of {% raw %}{% if page.slideshow %}{% endraw %} so I can enable it only if needed by putting `slideshow: true` into a posts YAML Front Matter. Have a look at [`_includes/slideshow2.html`](https://github.com/hummat/hummat.github.io/blob/master/_includes/slideshow2.html) instead if you need multiple slideshows.

- #### Adding videos

  Adding videos to your site is extremely easy. Simply add it to a directory inside your GitHub Pages repository, say `videos`, and use

  ```html
  <video width="100%" height="auto" loop autoplay controls>
    <source type="video/mp4" src="/videos/my_video.mp4" />
  </video>
  ```

  Note that, when working offline, there seems to be a bug with Chrome/Chromium (at the time of me writing this) where the video won't play. Don't worry, it will play online and you can still test it offline with another browser like Firefox.

- #### Interactive code

  Interactive figures are one thing, but as you most likely produced them using some code, wouldn’t it be nice if people also could interact with the _code_ and change the figure to their liking? Fortunately that’s super simple! Like already shown [here](#5-interactive-visualizations), you can create your figures in a [Jupyter notebook](https://jupyter.org/)—like [this one](https://github.com/hummat/hummat.github.io/blob/master/notebooks/beautiful-blogging.ipynb) which you can find in the [notebooks](https://github.com/hummat/hummat.github.io/blob/master/notebooks) directory—and add an [`environment.yml`](https://github.com/hummat/hummat.github.io/blob/master/environment.yml)[^9] file to the root of your GitHub page repository, filling in all packages you used in your notebook. Then head over to the [Binder website](https://mybinder.org/).

  [^9]: This assumes you are using `conda`, if you use `pip` or other languages than Python, check out the [Binder documentation](https://mybinder.readthedocs.io/en/latest/config_files.html).

  Over there, you simply fill in the URL of your [GitHub pages repository](https://github.com/hummat/hummat.github.io) and the path to the notebook you want to run—like `/notebooks/beautiful-blogging.ipynb`—and copy the markdown generated below. Paste it wherever you want to link to your code to get this: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/hummat/hummat.github.io/master?filepath=%2Fnotebooks%2Fbeautiful-blogging.ipynb)

  An interested reader can now simply click on it to start an interactive version of your code!

---
