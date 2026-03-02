# Infrastructure

## Overview

The site uses a split setup: Netlify builds and hosts, Cloudflare provides DNS and asset storage.

```text
GitHub (repo) → Netlify (build + CDN) → hummat.com
                Cloudflare (DNS) ──────→ routes traffic to Netlify
                Cloudflare R2 ─────────→ assets.hummat.com
```

## Services

| Service           | Role                                                                 | Dashboard           |
| ----------------- | -------------------------------------------------------------------- | ------------------- |
| **GitHub**        | Source repo (`hummat/hummat.github.io`)                              | github.com          |
| **Netlify**       | Jekyll build from `netlify` branch, CDN, SSL cert                    | app.netlify.com     |
| **Cloudflare**    | DNS (authoritative nameservers), R2 asset storage                    | dash.cloudflare.com |
| **Let's Encrypt** | Wildcard SSL cert (`*.hummat.com`, `hummat.com`), managed by Netlify | —                   |

## DNS (Cloudflare)

Nameservers: `sandy.ns.cloudflare.com`, `west.ns.cloudflare.com`

| Record            | Type  | Target                                   | Proxy            | Notes                                         |
| ----------------- | ----- | ---------------------------------------- | ---------------- | --------------------------------------------- |
| `hummat.com`      | CNAME | Netlify                                  | DNS only (grey)  | **Must** stay grey while on Netlify           |
| `www`             | CNAME | Netlify                                  | DNS only (grey)  | **Must** stay grey while on Netlify           |
| `assets`          | CNAME | R2 bucket                                | Proxied (orange) | Required for R2 custom domains                |
| `_acme-challenge` | CNAME | `_acme-challenge.hummat.com.netlify.com` | DNS only (grey)  | Delegates wildcard cert validation to Netlify |

### Why grey cloud for site records?

Cloudflare proxy (orange cloud) puts a CDN in front of Netlify's CDN. This causes:

- **SSL renewal failures** — Cloudflare terminates TLS, blocking Netlify's HTTP-01 validation
- **Double caching** — stale content after deploys
- **Header conflicts** — both providers add/modify response headers

Netlify's CDN already provides global edge delivery, DDoS protection, and HTTPS. The proxy adds no value here.

### Why orange cloud for assets?

Cloudflare R2 custom domains require the proxy to be enabled — it's how R2 serves content on your domain.

## SSL Certificate

- **Type:** Wildcard (`*.hummat.com`, `hummat.com`)
- **Provider:** Let's Encrypt, auto-renewed by Netlify
- **Validation:** DNS-01 via `_acme-challenge` CNAME delegation

If renewal fails, check:

1. `_acme-challenge` CNAME exists on Cloudflare (DNS only, not proxied)
2. Netlify DNS zone is **not** active (was deleted March 2026; a stale zone conflicts with Cloudflare)
3. Let's Encrypt rate limits haven't been hit (wait and retry)

```bash
# Verify ACME delegation is working
dig CNAME _acme-challenge.hummat.com +short
# Should return: _acme-challenge.hummat.com.netlify.com.
```

## Build & Deploy

- Netlify watches the `netlify` branch
- On push: runs `bundle exec jekyll build`, deploys to Netlify CDN
- Config: `netlify.toml` (build command, headers, redirects)
- PR previews: automatic deploy previews on PRs to `netlify`

## Planned Migration

Moving to Astro + Cloudflare Pages (see [#97](https://github.com/hummat/hummat.github.io/issues/97)). After migration:

- Remove `_acme-challenge` CNAME (Cloudflare Pages manages its own certs)
- Can re-enable proxy (orange cloud) for `hummat.com` and `www` (single CDN, no conflict)
- Delete Netlify site
