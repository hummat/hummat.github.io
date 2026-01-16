# Repository Guidelines

## Project Structure

- Jekyll site at repo root. Main pages: `index.md`, `about.md`, `publications.md`, `404.md`, `CNAME`, `_redirects`.
- Posts: `_posts/YYYY-MM-DD-kebab-title.md`. Drafts: `_drafts/`.
- Layouts & includes: `_layouts/`, `_includes/` (e.g. `default.html`, `home.html`, `post.html`, `popup.html`, `slideshow*.html`, `jquery.html`, `mathjax.html`).
- Styling: `_sass/` partials with `style.scss` as entrypoint.
- Assets & data: `images/`, `figures/` (HTML/Plotly), `data/` (numpy, ply, etc.), `notebooks/` (Jupyter). `_site/` is generated output; do not edit.

## Build & Serve

- Install: `bundle install`
- Serve: `bundle exec jekyll serve` (add `--drafts` for drafts)
- Build: `bundle exec jekyll build`
- Notebooks: `conda activate blog` (from `environment.yml`), then Jupyter

## Front Matter

```yaml
layout: post
title: My Title
abstract: Short summary
category: learning  # thought, resource, book, repository
tags: [tag1, tag2]
time: 8
words: 1500
plotly: true   # Plotly CDN
mathjax: true  # MathJax
jquery: true   # data-include loader
slideshow2: true
```

- `gh-page`: external project pages; `circular: true`: round thumbnails
- Interactive figures: notebooks write HTML to `figures/`, posts include via `<div data-include="https://assets.hummat.com/figures/….html"></div>`

## Style & Validation

- HTML/JS/SCSS: 2-space indent; prefer existing classes/partials
- Categories/tags: use current taxonomy (`learning`, `thought`, `resource`, `book`, `repository`)
- Validate: `bundle exec jekyll build`; for notebook posts, rerun notebooks first
- Commits: present tense (`Fix broken links`)
- PRs: describe change, screenshots for visual updates

## Tooling

- Linters: `npm run lint:md`, `npm run lint:scss`, `npm run lint:js`, `npm run format`
- Site check: `npm run test:site`
- Excluded from linters: `_site/`, `figures/`, `images/`, `data/`

## Assets

Assets stored in Cloudflare R2, uploaded via pre-commit hook. For asset-only uploads: `git commit --allow-empty -m "Upload new assets"`. See `ASSETS.md`.

## GitHub Issues & Labels

Templates and labels defined in `.github/`.

| Template | Purpose | Auto-label |
|----------|---------|------------|
| `post-idea.yml` | New post concepts | `type:idea` |
| `post-feedback.yml` | Errors/improvements for existing posts | `type:feedback` |
| `feature-request.yml` | Site functionality requests | `type:feature` |

Labels: `type:*` (primary), `category:*` (post category), `feedback:*` (error/clarity/addition), `area:*` (navigation/design/interactivity/performance), `status:*` (planned/in-progress).

Automation: `.github/workflows/issue-labeler.yml` adds secondary labels from form dropdowns. `.github/scripts/sync-labels.sh` syncs labels via `gh` CLI.

## New Post Workflow

1. Open Post Idea issue with title, category, description
2. Create branch: `git checkout -b post/<issue-number>-<slug>`
3. Write draft: `_drafts/<slug>.md`
4. Preview: `bundle exec jekyll serve --drafts`
5. Finalize: move to `_posts/YYYY-MM-DD-<slug>.md`
6. Open PR with `Closes #<issue-number>`
7. Merge — issue auto-closes

## Agent Notes

- `_config.yml` is single source of truth for site metadata, plugins, analytics, Disqus
- Don't edit `_site/`; change source files and let Jekyll regenerate
- Preserve behavior of JS helpers (`popup.html`, `slideshow*.html`, `jquery.html`)
- To track post ideas or features, create GitHub issue via `gh issue create`
