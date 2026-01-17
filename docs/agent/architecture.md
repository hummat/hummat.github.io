# Architecture

## Layout

```
.
├── CLAUDE.md                 # agent instructions (index)
├── _config.yml               # site config (single source of truth)
├── index.md                  # homepage
├── about.md, publications.md # main pages
├── _posts/                   # published posts (YYYY-MM-DD-slug.md)
├── _drafts/                  # unpublished drafts (slug.md)
├── _layouts/                 # page templates
│   ├── default.html          # base layout
│   ├── home.html             # homepage with post list
│   ├── post.html             # individual post
│   └── page.html             # static pages
├── _includes/                # reusable partials
│   ├── popup.html            # image popup JS
│   ├── slideshow*.html       # slideshow components
│   ├── jquery.html           # data-include loader
│   └── mathjax.html          # LaTeX rendering
├── _sass/                    # SCSS partials
├── style.scss                # main stylesheet (imports partials)
├── images/                   # local images (prefer R2 assets)
├── figures/                  # generated HTML (Plotly, etc.)
├── data/                     # numpy, ply, other data files
├── notebooks/                # Jupyter notebooks
└── _site/                    # generated output (DO NOT EDIT)
```

## Conventions

- **Posts**: `_posts/YYYY-MM-DD-kebab-title.md` with YAML front matter
- **Drafts**: `_drafts/slug.md` (no date prefix); preview with `--drafts`
- **Config**: `_config.yml` for site metadata, plugins, analytics, Disqus
- **Assets**: stored in Cloudflare R2; see `ASSETS.md`

## Key Files

| File | Purpose |
|------|---------|
| `_config.yml` | Site settings, plugin config, analytics IDs |
| `_layouts/home.html` | Post listing logic, category/tag display |
| `_layouts/post.html` | Single post template, metadata, comments |
| `style.scss` | All styling; imports `_sass/` partials |
| `_includes/jquery.html` | Enables `data-include` for external HTML |

## Adding Content

1. **New post**: create in `_drafts/`, move to `_posts/` when ready
2. **New page**: create `.md` at root with `layout: page`
3. **New include**: add to `_includes/`, use `{% include name.html %}`
4. **New style**: add partial to `_sass/`, import in `style.scss`
