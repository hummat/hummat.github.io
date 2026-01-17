# Content Guidelines

**Read before writing or editing posts.**

## Front Matter

```yaml
---
layout: post
title: Post Title
abstract: Short summary for listings and SEO
category: learning
tags: [machine-learning, python]
time: 8                # reading time in minutes
words: 1500            # approximate word count
image: https://assets.hummat.com/images/thumb.png  # thumbnail
---
```

### Optional Fields

| Field | Purpose |
|-------|---------|
| `gh-page` | External link (for `repository`/`paper` categories) |
| `circular: true` | Round thumbnail (logos, avatars) |
| `mathjax: true` | Enable LaTeX math rendering |
| `plotly: true` | Load Plotly CDN for charts |
| `jquery: true` | Enable `data-include` for external HTML |
| `slideshow2: true` | Enable slideshow component |
| `gradient: true` | Colored header with GitHub button |
| `github: <url>` | "View on GitHub" button in header |

## Categories

| Category | Description | External link? |
|----------|-------------|----------------|
| `learning` | Tutorial, educational content | rarely |
| `thought` | Personal reflection, opinion | no |
| `resource` | Curated list, reference guide | sometimes |
| `book` | Book summary or review | no |
| `repository` | Project showcase (GitHub repo) | yes (`gh-page`) |
| `paper` | Published paper page | yes (`gh-page`) |

## Tags

- Use existing tags when possible (check other posts)
- Lowercase, hyphenated: `machine-learning`, `point-clouds`
- Keep to 3-5 tags per post

## Interactive Content

### Plotly Charts

1. Generate HTML in notebook: `fig.write_html("figures/chart.html")`
2. Upload to R2 (via pre-commit hook)
3. Include in post:
   ```html
   <div data-include="https://assets.hummat.com/figures/chart.html"></div>
   ```
4. Enable in front matter: `jquery: true`

### Math (LaTeX)

Enable with `mathjax: true`, then use:
- Inline: `$E = mc^2$`
- Display: `$$\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}$$`

### Images

```markdown
![Alt text](https://assets.hummat.com/images/photo.jpg)
```

For popup behavior, images are automatically enhanced by `_includes/popup.html`.

## Style

- Write in clear, accessible prose
- Use headers (`##`, `###`) to structure content
- Code blocks with language hints: ` ```python `
- Keep paragraphs focused; break up long sections
