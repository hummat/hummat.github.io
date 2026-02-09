const test = require("node:test");
const assert = require("node:assert/strict");
const { createDom, runBrowserScript, flushMicrotasks } = require("./test-utils");

function createPublicationsDom() {
  return createDom({
    bodyHtml:
      '<a id="semantic-scholar-profile-link"></a>' +
      '<div id="publications-root" data-semantic-scholar-author-id="1753619041"></div>',
  });
}

test("publications renders sorted entries after successful fetch", async () => {
  const dom = createPublicationsDom();
  Object.defineProperty(dom.window.document, "readyState", {
    configurable: true,
    get: () => "complete",
  });
  dom.window.fetch = async () => ({
    status: 200,
    ok: true,
    headers: { get: () => null },
    json: async () => ({
      url: "/author/1753619041",
      papers: [
        {
          title: "Older paper",
          year: 2022,
          venue: "Venue A",
          citationCount: 1,
          url: "https://example.com/old",
          authors: [{ name: "Matthias Humt" }],
        },
        {
          title: "Newer paper",
          year: 2024,
          venue: "Venue B",
          citationCount: 2,
          url: "https://example.com/new",
          authors: [{ name: "Matthias Humt" }],
        },
      ],
    }),
  });

  runBrowserScript(dom, "assets/js/publications.js");
  await flushMicrotasks();

  const items = dom.window.document.querySelectorAll(".publication-item");
  assert.equal(items.length, 2);
  assert.match(items[0].textContent, /Newer paper/);
  assert.match(items[1].textContent, /Older paper/);

  const profileLink = dom.window.document.getElementById("semantic-scholar-profile-link");
  assert.match(profileLink.href, /semanticscholar\.org\/author\/1753619041\?utm_source=api/);
});

test("publications shows retry UI on fetch failure", async () => {
  const dom = createPublicationsDom();
  Object.defineProperty(dom.window.document, "readyState", {
    configurable: true,
    get: () => "complete",
  });
  dom.window.fetch = async () => {
    throw new Error("network failure");
  };
  dom.window.console.error = () => {};

  runBrowserScript(dom, "assets/js/publications.js");
  await flushMicrotasks(4);

  const root = dom.window.document.getElementById("publications-root");
  assert.match(root.textContent, /Failed to load publications from Semantic Scholar/i);
  assert.ok(root.querySelector(".publication-refresh-btn"));
});
