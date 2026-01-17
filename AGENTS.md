# Repository Guidelines

This file provides guidance to AI coding agents when working with this repository.

## Conventions

Read relevant `docs/agent/` files before proceeding:
- `workflow.md` — **read before starting any post or feature** (issues, branching, PRs)
- `content.md` — **read before writing posts** (front matter, categories, interactive content)
- `architecture.md` — read before modifying layouts/structure
- `tooling.md` — read before building, linting, or managing assets

**New posts/features:** Always read and follow `docs/agent/workflow.md` first. Create a labeled GitHub issue before implementing.

---

## Quick Reference

### Commands

```bash
# Dev
bundle install                      # install dependencies
bundle exec jekyll serve --drafts   # serve with drafts
bundle exec jekyll build            # build for production

# Lint
npm run lint:md                     # markdown
npm run lint:scss                   # styles
npm run format                      # format all
```

### Post Front Matter

```yaml
---
layout: post
title: Post Title
abstract: Short summary
category: learning  # thought, resource, book, repository, paper
tags: [tag1, tag2]
time: 5
words: 1000
---
```

### Style

- HTML/JS/SCSS: 2-space indent
- Commits: present tense (`Fix typo`, `Add new post`)
- Minimal diffs; match existing patterns

## Key Rules

- **Always follow `docs/agent/workflow.md`** — issue first, then branch, then PR
- **CRITICAL: Branch from `netlify`, PR to `netlify`** — NOT `main` (main is barebones for forks)
- `_config.yml` is single source of truth for site settings
- Don't edit `_site/`; change source files and let Jekyll regenerate
- Preserve behavior of JS helpers (`popup.html`, `slideshow*.html`, `jquery.html`)
- Assets go to Cloudflare R2, not the repo (see `ASSETS.md`)
