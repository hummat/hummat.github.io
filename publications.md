---
layout: page
title: Publications
---

<div id="publications-root"
     data-semantic-scholar-author-id="{{ site.semantic_scholar_author_id | escape }}">
</div>

<script>
(function() {
  function updateProfileLink(authorUrl) {
    var link = document.getElementById('semantic-scholar-profile-link');
    if (!link || !authorUrl) {
      return;
    }

    var href = authorUrl;
    if (href.indexOf('http') !== 0) {
      href = 'https://www.semanticscholar.org' + href;
    }
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
      container.innerHTML = '<p>Please set <code>semantic_scholar_author_id</code> in <code>_config.yml</code> to show publications.</p>';
      return;
    }

    var status = document.createElement('p');
    status.textContent = 'Loading publications from Semantic Scholar…';
    container.appendChild(status);

    var apiUrl = 'https://api.semanticscholar.org/graph/v1/author/' +
      encodeURIComponent(authorId) +
      '?fields=url,papers.title,papers.year,papers.venue,papers.citationCount,papers.url,papers.authors';

    function renderError(message) {
      status.textContent = message;
    }

    function renderPublications(data) {
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

      status.remove();

      var list = document.createElement('ol');
      list.className = 'publication-list';

    papers.forEach(function(paper) {
      if (!paper) return;

      var li = document.createElement('li');
      li.className = 'publication-item';

      var titleContainer = document.createElement('div');
      titleContainer.className = 'publication-title';

      var titleText = paper.title || 'Untitled';
      if (paper.url) {
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
      var authorsText = authors.map(function(a) { return a && a.name ? a.name : ''; })
        .filter(function(name) { return name; })
        .join(', ');

      if (authorsText) {
        var authorsEl = document.createElement('div');
        authorsEl.className = 'publication-authors';
        authorsEl.textContent = authorsText;
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

    if (!window.fetch) {
      renderError('Your browser is too old to load publications automatically.');
      return;
    }

    fetch(apiUrl)
      .then(function(response) {
        if (!response.ok) {
          throw new Error('Semantic Scholar API error: ' + response.status);
        }
        return response.json();
      })
      .then(function(data) {
        updateProfileLink(data && data.url);
        renderPublications(data);
      })
      .catch(function(err) {
        console.error(err);
        renderError('Failed to load publications from Semantic Scholar.');
      });
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
