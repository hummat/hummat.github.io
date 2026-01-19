# Welcome to the blog repo of [Matthias Humt](https://hummat.com)! [![Netlify Status](https://api.netlify.com/api/v1/badges/bf7cbae0-4ef7-4df1-b883-8863cbc3df09/deploy-status)](https://app.netlify.com/sites/hummat/deploys)

## Roadmap

See the [Blog Roadmap](https://github.com/users/hummat/projects/2) project for planned features and content.

## Branches

- **`netlify`** — production branch (deployed to [hummat.com](https://hummat.com))
- **`main`** — minimal template for forking (no posts)

If you like the design, fork [`main`](https://github.com/hummat/hummat.github.io/tree/main). To contribute, branch from `netlify`.

## Credits

Thanks to [**Barry Clark**](https://www.barryclark.com) and his wonderful [**Jekyll Now**](https://github.com/barryclark/jekyll-now) blog template!

## Tooling

- Install deps: `bundle install` and `npm install`
- Lint/format: `npm run lint:md`, `npm run lint:scss`, `npm run lint:js`, `npm run format`
- Build: `npm run test:site` (runs `bundle exec jekyll build`)
- Serve locally: `bundle exec jekyll serve [--drafts]`
