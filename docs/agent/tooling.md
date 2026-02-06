# Tooling

## Build & Serve

```bash
# Install dependencies
bundle install

# Serve locally (with live reload)
bundle exec jekyll serve

# Serve with drafts visible
bundle exec jekyll serve --drafts

# Build for production
bundle exec jekyll build
```

The site is served at `http://localhost:4000`. Output goes to `_site/` (do not edit directly).

## Linting & Formatting

```bash
# Markdown
npm run lint:md

# SCSS
npm run lint:scss

# JavaScript
npm run lint:js

# Format all (writes changes)
npm run format

# Check formatting (read-only, used in CI)
npm run format:check

# Build site
npm run test:site

# HTMLProofer (internal links only; requires build first)
npm run test:html
```

Excluded from linters: `_site/`, `figures/`, `images/`, `data/`

## Pre-commit Hooks

The repo uses `core.hooksPath = .githooks/`. On every commit:

1. **lint-staged** runs linters on staged files only (fast feedback)
2. **R2 upload** pushes any files in `_assets/` to Cloudflare R2

To set up hooks after cloning:

```bash
git config core.hooksPath .githooks
```

lint-staged config lives in `package.json` under the `"lint-staged"` key. It runs markdownlint, stylelint, eslint, and prettier on the relevant staged file types.

## CI Workflow

`.github/workflows/lint.yml` runs on PRs to `netlify`:

| Job     | What it does                                          |
| ------- | ----------------------------------------------------- |
| `lint`  | markdownlint, stylelint, eslint, prettier (Node-only) |
| `build` | Jekyll build + HTMLProofer (needs lint to pass first) |

Stale runs are auto-cancelled when a new push arrives (`cancel-in-progress`).

## Assets (Cloudflare R2)

Assets are stored in Cloudflare R2, not in the repo. The pre-commit hook handles uploads automatically.

```bash
# Upload assets with an otherwise empty commit
git commit --allow-empty -m "Upload new assets"
```

See `ASSETS.md` for full details on asset management.

## Notebooks

```bash
# Activate environment
conda activate blog  # from environment.yml

# Run Jupyter
jupyter notebook

# Or JupyterLab
jupyter lab
```

Notebooks in `notebooks/` generate HTML figures to `figures/`.

## GitHub Labels

```bash
# Sync labels from labels.yml to GitHub
.github/scripts/sync-labels.sh
```

## Validation Checklist

Before committing:

1. `bundle exec jekyll build` — no errors
2. `bundle exec jekyll serve` — visual check
3. For notebook posts: re-run notebook, regenerate figures
4. For style changes: check responsive behavior

## Common Issues

| Issue                         | Solution                                         |
| ----------------------------- | ------------------------------------------------ |
| Build fails with Liquid error | Check template syntax in `_includes/` or post    |
| Missing image                 | Verify R2 upload; check URL path                 |
| MathJax not rendering         | Add `mathjax: true` to front matter              |
| Plotly chart not loading      | Add `jquery: true` and verify `data-include` URL |
