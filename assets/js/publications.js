(function () {
  "use strict";

  const CACHE_TTL = 24 * 60 * 60 * 1000; // 24 hours
  const MAX_RETRIES = 3;
  const BASE_DELAY = 1000; // 1 second

  function getCacheKey(authorId) {
    return `ss-pubs-${authorId}`;
  }

  function readCache(authorId) {
    try {
      const raw = localStorage.getItem(getCacheKey(authorId));
      if (!raw) {
        return null;
      }
      return JSON.parse(raw);
    } catch (_err) {
      return null;
    }
  }

  function writeCache(authorId, data) {
    try {
      localStorage.setItem(
        getCacheKey(authorId),
        JSON.stringify({
          data,
          timestamp: Date.now(),
        })
      );
    } catch (_err) {
      // Quota exceeded or unavailable.
    }
  }

  function isCacheFresh(cached) {
    return cached && Date.now() - cached.timestamp < CACHE_TTL;
  }

  function isHttpUrl(url) {
    try {
      const parsed = new URL(url);
      return parsed.protocol === "http:" || parsed.protocol === "https:";
    } catch (_err) {
      return false;
    }
  }

  function updateProfileLink(authorUrl) {
    const link = document.getElementById("semantic-scholar-profile-link");
    if (!link || !authorUrl) {
      return;
    }

    let href = authorUrl;
    if (!href.startsWith("http")) {
      href = `https://www.semanticscholar.org${href}`;
    }
    if (!isHttpUrl(href)) {
      return;
    }

    link.href = href + (href.includes("?") ? "&utm_source=api" : "?utm_source=api");
  }

  function init() {
    const container = document.getElementById("publications-root");
    if (!container) {
      return;
    }

    let authorId = container.getAttribute("data-semantic-scholar-author-id") || "";
    authorId = authorId.trim();

    if (!authorId || authorId === "REPLACE_WITH_YOUR_SEMANTIC_SCHOLAR_AUTHOR_ID") {
      container.textContent = "";
      const msg = document.createElement("p");
      const code1 = document.createElement("code");
      code1.textContent = "semantic_scholar_author_id";
      const code2 = document.createElement("code");
      code2.textContent = "_config.yml";
      msg.appendChild(document.createTextNode("Please set "));
      msg.appendChild(code1);
      msg.appendChild(document.createTextNode(" in "));
      msg.appendChild(code2);
      msg.appendChild(document.createTextNode(" to show publications."));
      container.appendChild(msg);
      return;
    }

    let status = document.createElement("p");
    status.textContent = "Loading publications from Semantic Scholar…";
    container.appendChild(status);

    const apiUrl =
      `https://api.semanticscholar.org/graph/v1/author/${encodeURIComponent(authorId)}` +
      "?fields=url,papers.title,papers.year,papers.venue,papers.citationCount,papers.url,papers.authors";

    function clearContainer() {
      while (container.firstChild) {
        container.removeChild(container.firstChild);
      }
    }

    function renderError(message, options) {
      const opts = options || {};
      clearContainer();

      const p = document.createElement("p");
      p.textContent = message;
      container.appendChild(p);

      if (opts.showRetry) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "publication-refresh-btn";
        btn.textContent = "Try again";
        btn.addEventListener("click", () => {
          clearContainer();
          status = document.createElement("p");
          status.textContent = "Loading publications from Semantic Scholar…";
          container.appendChild(status);
          fetchWithRetry(0);
        });
        container.appendChild(btn);
      }
    }

    function renderAuthorNames(authors, parentEl) {
      for (let i = 0; i < authors.length; i += 1) {
        const author = authors[i];
        const name = author && author.name ? author.name : "";
        if (!name) {
          continue;
        }

        if (i > 0) {
          parentEl.appendChild(document.createTextNode(", "));
        }

        if (name === "Matthias Humt") {
          const strong = document.createElement("strong");
          strong.textContent = name;
          parentEl.appendChild(strong);
        } else {
          parentEl.appendChild(document.createTextNode(name));
        }
      }
    }

    function renderPublications(data, options) {
      const opts = options || {};
      if (!data || !Array.isArray(data.papers) || data.papers.length === 0) {
        renderError("No publications found for this Semantic Scholar author ID.");
        return;
      }

      const papers = data.papers.slice().sort((a, b) => {
        const ay = a.year || 0;
        const by = b.year || 0;
        if (ay !== by) {
          return by - ay;
        }
        const at = (a.title || "").toLowerCase();
        const bt = (b.title || "").toLowerCase();
        if (at < bt) {
          return -1;
        }
        if (at > bt) {
          return 1;
        }
        return 0;
      });

      clearContainer();

      if (opts.cached) {
        const notice = document.createElement("p");
        notice.className = "publication-cache-notice";
        notice.appendChild(document.createTextNode("Showing cached data · "));

        const refreshLink = document.createElement("button");
        refreshLink.type = "button";
        refreshLink.className = "publication-refresh-btn";
        refreshLink.textContent = "Refresh";
        refreshLink.addEventListener("click", () => {
          clearContainer();
          status = document.createElement("p");
          status.textContent = "Loading publications from Semantic Scholar…";
          container.appendChild(status);
          fetchWithRetry(0);
        });

        notice.appendChild(refreshLink);
        container.appendChild(notice);
      }

      const list = document.createElement("ol");
      list.className = "publication-list";

      papers.forEach((paper) => {
        if (!paper) {
          return;
        }

        const li = document.createElement("li");
        li.className = "publication-item";

        const titleContainer = document.createElement("div");
        titleContainer.className = "publication-title";

        const titleText = paper.title || "Untitled";
        if (paper.url && isHttpUrl(paper.url)) {
          const link = document.createElement("a");
          link.href = paper.url + (paper.url.includes("?") ? "&utm_source=api" : "?utm_source=api");
          link.textContent = titleText;
          link.target = "_blank";
          link.rel = "noopener noreferrer";
          titleContainer.appendChild(link);
        } else {
          titleContainer.textContent = titleText;
        }

        li.appendChild(titleContainer);

        const authors = Array.isArray(paper.authors) ? paper.authors : [];
        const hasAuthors = authors.some((author) => author && author.name);

        if (hasAuthors) {
          const authorsEl = document.createElement("div");
          authorsEl.className = "publication-authors";
          renderAuthorNames(authors, authorsEl);
          li.appendChild(authorsEl);
        }

        const metaParts = [];
        if (paper.year) {
          metaParts.push(String(paper.year));
        }
        if (paper.venue) {
          metaParts.push(paper.venue);
        }
        if (typeof paper.citationCount === "number") {
          metaParts.push(`${paper.citationCount} citation${paper.citationCount === 1 ? "" : "s"}`);
        }

        if (metaParts.length > 0) {
          const metaEl = document.createElement("div");
          metaEl.className = "publication-meta";
          metaEl.textContent = metaParts.join(" · ");
          li.appendChild(metaEl);
        }

        list.appendChild(li);
      });

      container.appendChild(list);
    }

    function handleFetchFailure(error) {
      console.error(error);
      const cached = readCache(authorId);
      if (cached && cached.data) {
        updateProfileLink(cached.data.url);
        renderPublications(cached.data, { cached: true });
      } else {
        renderError("Failed to load publications from Semantic Scholar.", { showRetry: true });
      }
    }

    function fetchWithRetry(attempt) {
      const controller = typeof AbortController !== "undefined" ? new AbortController() : null;
      let timeoutId;

      const fetchPromise = fetch(apiUrl, controller ? { signal: controller.signal } : {});

      const timeoutPromise = new Promise((_, reject) => {
        timeoutId = setTimeout(() => {
          if (controller) {
            controller.abort();
          }
          reject(new Error("Request timed out"));
        }, 15000);
      });

      return Promise.race([fetchPromise, timeoutPromise])
        .then((response) => {
          if (response.status === 429 && attempt < MAX_RETRIES) {
            const retryAfter = Number.parseInt(response.headers.get("Retry-After"), 10);
            const delay =
              Number.isFinite(retryAfter) && retryAfter > 0
                ? retryAfter * 1000
                : BASE_DELAY * 2 ** attempt;

            return new Promise((resolve) => {
              setTimeout(resolve, delay);
            }).then(() => fetchWithRetry(attempt + 1));
          }

          if (!response.ok) {
            throw new Error(`Semantic Scholar API error: ${response.status}`);
          }

          return response.json();
        })
        .catch((error) => {
          handleFetchFailure(error);
          return null;
        })
        .then((data) => {
          if (!data) {
            return;
          }

          writeCache(authorId, data);
          updateProfileLink(data && data.url);
          renderPublications(data);
        })
        .finally(() => {
          clearTimeout(timeoutId);
        });
    }

    if (!window.fetch) {
      renderError("Your browser is too old to load publications automatically.");
      return;
    }

    const cached = readCache(authorId);
    if (isCacheFresh(cached)) {
      updateProfileLink(cached.data && cached.data.url);
      renderPublications(cached.data);
      return;
    }

    fetchWithRetry(0);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
