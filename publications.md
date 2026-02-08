---
layout: page
title: Publications
---

<div id="publications-root"
     data-semantic-scholar-author-id="{{ site.semantic_scholar_author_id | escape }}">
</div>

<script>
(function() {
  var CACHE_TTL = 24 * 60 * 60 * 1000; // 24 hours
  var MAX_RETRIES = 3;
  var BASE_DELAY = 1000; // 1 second

  function getCacheKey(authorId) {
    return 'ss-pubs-' + authorId;
  }

  function readCache(authorId) {
    try {
      var raw = localStorage.getItem(getCacheKey(authorId));
      if (!raw) return null;
      return JSON.parse(raw);
    } catch (e) {
      return null;
    }
  }

  function writeCache(authorId, data) {
    try {
      localStorage.setItem(getCacheKey(authorId), JSON.stringify({
        data: data,
        timestamp: Date.now()
      }));
    } catch (e) {
      // Quota exceeded or unavailable — ignore
    }
  }

  function isCacheFresh(cached) {
    return cached && (Date.now() - cached.timestamp) < CACHE_TTL;
  }

  function isHttpUrl(url) {
    try {
      var parsed = new URL(url);
      return parsed.protocol === 'http:' || parsed.protocol === 'https:';
    } catch (e) {
      return false;
    }
  }

  function updateProfileLink(authorUrl) {
    var link = document.getElementById('semantic-scholar-profile-link');
    if (!link || !authorUrl) {
      return;
    }

    var href = authorUrl;
    if (href.indexOf('http') !== 0) {
      href = 'https://www.semanticscholar.org' + href;
    }
    if (!isHttpUrl(href)) return;
    link.href = href + (href.indexOf('?') === -1 ? '?utm_source=api' : '&utm_source=api');
  }

  function init() {
    var container = document.getElementById('publications-root');
    if (!container) {
      return;
    }

    var authorId = container.getAttribute('data-semantic-scholar-author-id') || '';
    authorId = authorId.trim();

    if (!authorId || authorId === 'REPLACE_WITH_YOUR_SEMANTIC_SCHOLAR_AUTHOR_ID') {
      container.textContent = '';
      var msg = document.createElement('p');
      var code1 = document.createElement('code');
      code1.textContent = 'semantic_scholar_author_id';
      var code2 = document.createElement('code');
      code2.textContent = '_config.yml';
      msg.appendChild(document.createTextNode('Please set '));
      msg.appendChild(code1);
      msg.appendChild(document.createTextNode(' in '));
      msg.appendChild(code2);
      msg.appendChild(document.createTextNode(' to show publications.'));
      container.appendChild(msg);
      return;
    }

    var status = document.createElement('p');
    status.textContent = 'Loading publications from Semantic Scholar…';
    container.appendChild(status);

    var apiUrl = 'https://api.semanticscholar.org/graph/v1/author/' +
      encodeURIComponent(authorId) +
      '?fields=url,papers.title,papers.year,papers.venue,papers.citationCount,papers.url,papers.authors';

    function clearContainer() {
      while (container.firstChild) {
        container.removeChild(container.firstChild);
      }
    }

    function renderError(message, options) {
      options = options || {};
      clearContainer();
      var p = document.createElement('p');
      p.textContent = message;
      container.appendChild(p);

      if (options.showRetry) {
        var btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'publication-refresh-btn';
        btn.textContent = 'Try again';
        btn.addEventListener('click', function() {
          clearContainer();
          status = document.createElement('p');
          status.textContent = 'Loading publications from Semantic Scholar…';
          container.appendChild(status);
          fetchWithRetry(0);
        });
        container.appendChild(btn);
      }
    }

    function renderAuthorNames(authors, parentEl) {
      // Build author list using DOM nodes to avoid innerHTML.
      // Bold the site owner's name; separate with ", ".
      for (var i = 0; i < authors.length; i++) {
        var a = authors[i];
        var name = a && a.name ? a.name : '';
        if (!name) continue;
        if (i > 0) {
          parentEl.appendChild(document.createTextNode(', '));
        }
        if (name === 'Matthias Humt') {
          var strong = document.createElement('strong');
          strong.textContent = name;
          parentEl.appendChild(strong);
        } else {
          parentEl.appendChild(document.createTextNode(name));
        }
      }
    }

    function renderPublications(data, options) {
      options = options || {};
      if (!data || !Array.isArray(data.papers) || data.papers.length === 0) {
        renderError('No publications found for this Semantic Scholar author ID.');
        return;
      }

      var papers = data.papers.slice().sort(function(a, b) {
        var ay = a.year || 0;
        var by = b.year || 0;
        if (ay !== by) {
          return by - ay;
        }
        var at = (a.title || '').toLowerCase();
        var bt = (b.title || '').toLowerCase();
        if (at < bt) return -1;
        if (at > bt) return 1;
        return 0;
      });

      clearContainer();

      if (options.cached) {
        var notice = document.createElement('p');
        notice.className = 'publication-cache-notice';
        notice.appendChild(document.createTextNode('Showing cached data · '));
        var refreshLink = document.createElement('button');
        refreshLink.type = 'button';
        refreshLink.className = 'publication-refresh-btn';
        refreshLink.textContent = 'Refresh';
        refreshLink.addEventListener('click', function() {
          clearContainer();
          status = document.createElement('p');
          status.textContent = 'Loading publications from Semantic Scholar…';
          container.appendChild(status);
          fetchWithRetry(0);
        });
        notice.appendChild(refreshLink);
        container.appendChild(notice);
      }

      var list = document.createElement('ol');
      list.className = 'publication-list';

    papers.forEach(function(paper) {
      if (!paper) return;

      var li = document.createElement('li');
      li.className = 'publication-item';

      var titleContainer = document.createElement('div');
      titleContainer.className = 'publication-title';

      var titleText = paper.title || 'Untitled';
      if (paper.url && isHttpUrl(paper.url)) {
        var link = document.createElement('a');
        link.href = paper.url + (paper.url.indexOf('?') === -1 ? '?utm_source=api' : '&utm_source=api');
        link.textContent = titleText;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        titleContainer.appendChild(link);
      } else {
        titleContainer.textContent = titleText;
      }

      li.appendChild(titleContainer);

      var authors = Array.isArray(paper.authors) ? paper.authors : [];
      var hasAuthors = authors.some(function(a) { return a && a.name; });

      if (hasAuthors) {
        var authorsEl = document.createElement('div');
        authorsEl.className = 'publication-authors';
        renderAuthorNames(authors, authorsEl);
        li.appendChild(authorsEl);
      }

      var metaParts = [];
      if (paper.year) {
        metaParts.push(String(paper.year));
      }
      if (paper.venue) {
        metaParts.push(paper.venue);
      }
      if (typeof paper.citationCount === 'number') {
        metaParts.push(paper.citationCount + ' citation' + (paper.citationCount === 1 ? '' : 's'));
      }

      if (metaParts.length > 0) {
        var metaEl = document.createElement('div');
        metaEl.className = 'publication-meta';
        metaEl.textContent = metaParts.join(' · ');
        li.appendChild(metaEl);
      }

      list.appendChild(li);
    });

      container.appendChild(list);
    }

    function handleFetchFailure(err) {
      console.error(err);
      var cached = readCache(authorId);
      if (cached && cached.data) {
        updateProfileLink(cached.data.url);
        renderPublications(cached.data, { cached: true });
      } else {
        renderError('Failed to load publications from Semantic Scholar.', { showRetry: true });
      }
    }

    function fetchWithRetry(attempt) {
      var controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
      var timeoutId;

      var fetchPromise = fetch(apiUrl, controller ? { signal: controller.signal } : {});

      var timeoutPromise = new Promise(function(_, reject) {
        timeoutId = setTimeout(function() {
          if (controller) controller.abort();
          reject(new Error('Request timed out'));
        }, 15000);
      });

      return Promise.race([fetchPromise, timeoutPromise])
        .then(function(response) {
          clearTimeout(timeoutId);
          if (response.status === 429 && attempt < MAX_RETRIES) {
            var retryAfter = parseInt(response.headers.get('Retry-After'), 10);
            var delay = (retryAfter && retryAfter > 0)
              ? retryAfter * 1000
              : BASE_DELAY * Math.pow(2, attempt);
            return new Promise(function(resolve) {
              setTimeout(resolve, delay);
            }).then(function() {
              return fetchWithRetry(attempt + 1);
            });
          }
          if (!response.ok) {
            throw new Error('Semantic Scholar API error: ' + response.status);
          }
          return response.json().then(function(data) {
            writeCache(authorId, data);
            updateProfileLink(data && data.url);
            renderPublications(data);
          });
        })
        .catch(function(err) {
          clearTimeout(timeoutId);
          handleFetchFailure(err);
        });
    }

    if (!window.fetch) {
      renderError('Your browser is too old to load publications automatically.');
      return;
    }

    // Check cache first
    var cached = readCache(authorId);
    if (isCacheFresh(cached)) {
      updateProfileLink(cached.data && cached.data.url);
      renderPublications(cached.data);
      return;
    }

    fetchWithRetry(0);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
</script>

<p>
This page lists publications pulled automatically from
<a id="semantic-scholar-profile-link" href="https://www.semanticscholar.org/?utm_source=api">Semantic Scholar</a>.
</p>

<noscript>
  <p><strong>JavaScript is disabled.</strong> Enable it to load publications from Semantic Scholar.</p>
</noscript>
