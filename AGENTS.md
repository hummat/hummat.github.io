# Repository Guidelines

This file provides guidance to Claude Code (claude.ai/code) and other agents when working with code in this repository.

## Project Structure & Content

- Jekyll site at repo root. Main pages: `index.md`, `about.md`, `publications.md`, `404.md`, `CNAME`, `_redirects`.
- Posts: `_posts/YYYY-MM-DD-kebab-title.md`. Drafts: `_drafts/`.
- Layouts & includes: `_layouts/`, `_includes/` (e.g. `default.html`, `home.html`, `post.html`, `popup.html`, `slideshow*.html`, `jquery.html`, `mathjax.html`).
- Styling: `_sass/` partials with `style.scss` as entrypoint.
- Assets & data: `images/`, `figures/` (HTML/Plotly), `data/` (numpy, ply, etc.), `notebooks/` (Jupyter). `_site/` is generated output; do not edit.

## Build, Serve, and Notebooks

- Install Ruby deps: `bundle install`.
- Local dev: `bundle exec jekyll serve` (add `--drafts` while editing drafts).
- Build only: `bundle exec jekyll build`.
- Notebook env: `conda env create -f environment.yml`, then `conda activate blog` and run Jupyter. Use this env when regenerating figures or HTML.

## Content Conventions & Features

- Front matter example:

  ```yaml
  layout: post
  title: My Title
  abstract: Short summary
  category: learning
  tags: [tag1, tag2]
  time: 8
  words: 1500
  plotly: true  # load Plotly CDN
  mathjax: true # load MathJax
  jquery: true  # enable [data-include] loader
  slideshow2: true
  ```

- Use `gh-page` for external project pages and `circular: true` for round thumbnails.
- For interactive figures, notebooks should write HTML into `figures/` (or `_includes/figures/`) and posts should pull them in via `<div data-include="https://assets.hummat.com/figures/…html"></div>`.

## Style, Validation, and PRs

- HTML/JS/SCSS: 2‑space indentation; prefer existing classes/partials over new globals.
- Keep categories/tags aligned with current taxonomy (`learning`, `thought`, `resource`, `repository`, etc.).
- Validate changes with `bundle exec jekyll build`; for notebook-driven posts, rerun notebooks and regenerate dependent assets.
- Commit messages: short, present tense (e.g. `Fix broken links`, `Update banner image`).
- PRs: clearly describe the change, mention build status, and add screenshots for visual updates.

## Tooling

- Node-based linters live in `package.json`: `npm run lint:md` (markdownlint), `npm run lint:scss` (stylelint), `npm run lint:js` (eslint for inline scripts), `npm run format` (prettier).
- Site check: `npm run test:site` → `bundle exec jekyll build && bundle exec htmlproofer ./_site --check-html --check-opengraph --assume-extension`.
- Ignore generated output: `_site/`, `figures/`, `images/`, `data/`, and large binaries are excluded from formatters/linters via ignore files.

## Large File Storage

Assets are stored in Cloudflare R2. See [ASSETS.md](ASSETS.md) for details on uploading and referencing files.

## Agent-Specific Notes

- Treat `_config.yml` as the single source of truth for site metadata, plugins, analytics, Disqus, and redirects includes.
- Do not rewrite `_site/`; instead, change source Markdown, layouts, includes, or notebooks and let Jekyll regenerate.
- When editing JavaScript helpers (`popup.html`, `slideshow*.html`, `jquery.html`), preserve current behavior (footnote popups, slideshows, data-include loading) to avoid breaking existing posts.
