# Contributing

Thanks for your interest in contributing! This guide covers how to run the site locally and submit changes.

## Local Setup

### Prerequisites

- Ruby (see `.ruby-version`)
- Bundler (`gem install bundler`)
- Node.js (for linting, optional)

### Quick Start

```bash
# Install dependencies
bundle install

# Serve locally with live reload
bundle exec jekyll serve

# Include drafts
bundle exec jekyll serve --drafts
```

The site will be available at `http://localhost:4000`.

## Types of Contributions

### Content Fixes

Typos, broken links, factual errors, or unclear explanations in existing posts.

### New Posts

Follow this workflow for new posts:

1. **Open an issue** using the [Post Idea](https://github.com/hummat/hummat.github.io/issues/new?template=post-idea.yml) template
2. **Create a branch** from `netlify`:
   ```bash
   git checkout -b post/<issue-number>-<slug>
   # e.g., post/42-graph-neural-networks
   ```
3. **Write your draft** in `_drafts/<slug>.md`
4. **Preview locally**:
   ```bash
   bundle exec jekyll serve --drafts
   ```
5. **Finalize** by moving to `_posts/YYYY-MM-DD-<slug>.md` when ready
6. **Open a PR** with `Closes #<issue-number>` in the description
7. **Merge** — the issue will auto-close

#### Front Matter Template

```yaml
---
layout: post
title: Your Title
abstract: A short summary
category: learning  # or: thought, resource, book, repository, paper
tags: [tag1, tag2]
time: 5            # estimated reading time in minutes
words: 1000        # approximate word count
mathjax: true      # enable LaTeX math (optional)
plotly: true       # enable Plotly charts (optional)
---
```

#### Categories

- `learning` — tutorials, educational content
- `thought` — personal reflections, opinions
- `resource` — curated lists, reference guides
- `book` — book summaries or reviews
- `repository` — project showcases
- `paper` — published paper pages (use `gh-page` for external link)

### Site Improvements

Changes to layouts, styling, or functionality. For larger features, please [open an issue](https://github.com/hummat/hummat.github.io/issues/new/choose) first to discuss the approach.

## Code Style

- **Markdown**: Follow existing post formatting
- **HTML/JS/SCSS**: 2-space indentation
- **Commit messages**: Short, present tense ("Fix typo", "Add new post")

## Validation

Before submitting:

```bash
# Build the site (catches errors)
bundle exec jekyll build

# Optional: run linters
npm run lint:md
npm run lint:scss
```

## Pull Request Process

1. Fork the repository
2. Create a branch (`git checkout -b fix/typo-in-post`)
3. Make your changes
4. Test locally with `bundle exec jekyll serve`
5. Submit a PR with a clear description

## Issue Templates

Before contributing, consider opening an issue to discuss your idea:

| Template | Use for |
|----------|---------|
| [Post Idea](https://github.com/hummat/hummat.github.io/issues/new?template=post-idea.yml) | Suggest a topic for a new blog post |
| [Post Feedback](https://github.com/hummat/hummat.github.io/issues/new?template=post-feedback.yml) | Report errors or suggest improvements to existing posts |
| [Feature Request](https://github.com/hummat/hummat.github.io/issues/new?template=feature-request.yml) | Suggest site improvements (search, navigation, etc.) |

## Questions?

- Open an [issue](https://github.com/hummat/hummat.github.io/issues/new/choose) for bugs or suggestions
- Check existing issues before creating a new one
