# Post & Feature Workflow

**Read this file before starting any post or feature.**

## New Posts

1. **Create GitHub issue** — use the [Post Idea](https://github.com/hummat/hummat.github.io/issues/new?template=post-idea.yml) template; select appropriate category
2. **Create branch** — `git checkout -b post/<issue-number>-<slug>` (e.g., `post/42-graph-neural-networks`)
3. **Write draft** — create `_drafts/<slug>.md` with proper front matter
4. **Preview** — `bundle exec jekyll serve --drafts`
5. **Finalize** — move to `_posts/YYYY-MM-DD-<slug>.md`
6. **Create PR** — use the PR template, include `Closes #<issue-number>`
7. **Merge** — issue auto-closes

## Site Features

1. **Create GitHub issue** — use the [Feature Request](https://github.com/hummat/hummat.github.io/issues/new?template=feature-request.yml) template
2. **Create branch** — use appropriate prefix (see below)
3. **Implement** — follow code conventions
4. **Create PR** — use the PR template, reference issue

## Trivial Changes

Skip issue for typos, small fixes, docs-only changes. Branch + PR still recommended.

## Branch Naming

- `post/<issue>-<slug>` — new posts
- `feat/<name>` — new features
- `fix/<name>` — bug fixes
- `docs/<name>` — documentation only

## Pull Requests

**IMPORTANT:** Always open PRs against the `netlify` branch, not `main`.

- `netlify` — deploy branch with full site (Netlify builds from here)
- `main` — barebones version for forks (not for PRs)

```bash
# Create PR targeting netlify
gh pr create --base netlify --title "..." --body "..."
```

## Categories

| Category | Description | Use `gh-page` for external link |
|----------|-------------|--------------------------------|
| `learning` | Tutorial, educational content | rarely |
| `thought` | Personal reflection, opinion | no |
| `resource` | Curated list, reference guide | sometimes |
| `book` | Book summary or review | no |
| `repository` | Project showcase (GitHub repo) | yes |
| `paper` | Published paper page | yes |

For `repository` and `paper` posts, use `gh-page: <url>` in front matter to link directly to the external GitHub Pages site.

## Templates

- **Issues**: `.github/ISSUE_TEMPLATE/` (post-idea.yml, post-feedback.yml, feature-request.yml)
- **PRs**: `.github/PULL_REQUEST_TEMPLATE.md` — fill out Summary, Type, Checklist
- **Contributing**: `.github/CONTRIBUTING.md` for full setup guide

## Front Matter Reference

```yaml
---
layout: post
title: Post Title
abstract: Short summary for listings
category: learning  # learning, thought, resource, book, repository, paper
tags: [tag1, tag2]
time: 5             # reading time in minutes
words: 1000         # word count
image: https://assets.hummat.com/images/thumb.png  # thumbnail
gh-page: https://hummat.github.io/project  # external link (repository/paper)
circular: true      # round thumbnail (optional)
mathjax: true       # LaTeX math (optional)
plotly: true        # Plotly charts (optional)
jquery: true        # data-include loader (optional)
---
```
