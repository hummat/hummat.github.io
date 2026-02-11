const test = require("node:test");
const assert = require("node:assert/strict");
const { createDom, runBrowserScript, flushMicrotasks } = require("./test-utils");

function commentsFixtureHtml(provider = "giscus") {
  if (provider === "giscus") {
    return (
      '<div class="comments" data-comments-provider="giscus">' +
      '<div id="comments-consent-gate">' +
      '<button type="button" data-comments-action="accept">Load comments</button>' +
      "</div>" +
      '<p id="comments-consent-manage" hidden>' +
      '<button type="button" data-comments-action="revoke">Disable comments</button>' +
      "</p>" +
      '<div id="giscus_thread" data-giscus-repo="hummat/hummat.github.io" data-giscus-repo-id="MDEwOlJlcG9zaXRvcnkyNjM2MDE4OTY=" data-giscus-category="General" data-giscus-category-id="DIC_kwDOD7Y-6M4C1Coy" data-giscus-mapping="pathname" data-giscus-strict="0" data-giscus-reactions-enabled="1" data-giscus-emit-metadata="0" data-giscus-input-position="top" data-giscus-theme="preferred_color_scheme" data-giscus-lang="en" hidden></div>' +
      "</div>"
    );
  }

  if (provider === "utterances") {
    return (
      '<div class="comments" data-comments-provider="utterances">' +
      '<div id="comments-consent-gate">' +
      '<button type="button" data-comments-action="accept">Load comments</button>' +
      "</div>" +
      '<p id="comments-consent-manage" hidden>' +
      '<button type="button" data-comments-action="revoke">Disable comments</button>' +
      "</p>" +
      '<div id="utterances_thread" data-utterances-repo="hummat/hummat.github.io" data-utterances-issue-term="pathname" data-utterances-label="comments" data-utterances-theme="preferred-color-scheme" hidden></div>' +
      "</div>"
    );
  }

  return (
    '<div class="comments" data-comments-provider="disqus">' +
    '<div id="comments-consent-gate">' +
    '<button type="button" data-comments-action="accept">Load comments</button>' +
    "</div>" +
    '<p id="comments-consent-manage" hidden>' +
    '<button type="button" data-comments-action="revoke">Disable comments</button>' +
    "</p>" +
    '<div id="disqus_thread" data-disqus-shortname="example-shortname" hidden></div>' +
    "</div>"
  );
}

test("cookie banner controls analytics only", async () => {
  const dom = createDom({
    headHtml: '<meta name="google-analytics-id" content="G-TEST1234">',
    bodyHtml:
      '<div id="cookie-consent-banner" hidden>' +
      '<button type="button" data-consent-action="accept">Accept</button>' +
      '<button type="button" data-consent-action="decline">Decline</button>' +
      "</div>" +
      commentsFixtureHtml("giscus"),
  });
  Object.defineProperty(dom.window.document, "readyState", {
    configurable: true,
    get: () => "complete",
  });

  runBrowserScript(dom, "assets/js/privacy-consent.js");
  await flushMicrotasks();

  const banner = dom.window.document.getElementById("cookie-consent-banner");
  assert.equal(banner.hidden, false);
  assert.equal(dom.window.document.getElementById("ga4-loader"), null);
  assert.equal(dom.window.document.getElementById("giscus-embed-loader"), null);

  const acceptButton = banner.querySelector('[data-consent-action="accept"]');
  acceptButton.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
  await flushMicrotasks();

  assert.equal(dom.window.localStorage.getItem("hummat-cookie-consent-v1"), "accepted");

  const gaScript = dom.window.document.getElementById("ga4-loader");
  const giscusScript = dom.window.document.getElementById("giscus-embed-loader");
  assert.ok(gaScript);
  assert.equal(giscusScript, null);
  assert.match(gaScript.src, /googletagmanager\.com/);
  assert.equal(banner.hidden, true);
});

test("giscus comments load only after explicit click and persist preference", async () => {
  const dom = createDom({
    bodyHtml: '<div id="cookie-consent-banner" hidden></div>' + commentsFixtureHtml("giscus"),
  });
  Object.defineProperty(dom.window.document, "readyState", {
    configurable: true,
    get: () => "complete",
  });

  runBrowserScript(dom, "assets/js/privacy-consent.js");
  await flushMicrotasks();

  const gate = dom.window.document.getElementById("comments-consent-gate");
  const manage = dom.window.document.getElementById("comments-consent-manage");
  const thread = dom.window.document.getElementById("giscus_thread");

  assert.equal(gate.hidden, false);
  assert.equal(manage.hidden, true);
  assert.equal(thread.hidden, true);
  assert.equal(dom.window.document.getElementById("giscus-embed-loader"), null);

  const loadButton = gate.querySelector('[data-comments-action="accept"]');
  loadButton.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
  await flushMicrotasks();

  const giscusScript = dom.window.document.getElementById("giscus-embed-loader");
  assert.ok(giscusScript);
  assert.equal(dom.window.localStorage.getItem("hummat-comments-consent-v1"), "accepted");
  assert.equal(gate.hidden, true);
  assert.equal(manage.hidden, false);
  assert.equal(thread.hidden, false);
  assert.equal(giscusScript.getAttribute("data-repo"), "hummat/hummat.github.io");
  assert.equal(giscusScript.getAttribute("data-repo-id"), "MDEwOlJlcG9zaXRvcnkyNjM2MDE4OTY=");
  assert.equal(giscusScript.getAttribute("data-category"), "General");
  assert.equal(giscusScript.getAttribute("data-category-id"), "DIC_kwDOD7Y-6M4C1Coy");
  assert.equal(giscusScript.getAttribute("data-mapping"), "pathname");
  assert.equal(giscusScript.getAttribute("data-theme"), "preferred_color_scheme");
});

test("utterances comments load only after explicit click", async () => {
  const dom = createDom({
    bodyHtml: '<div id="cookie-consent-banner" hidden></div>' + commentsFixtureHtml("utterances"),
  });
  Object.defineProperty(dom.window.document, "readyState", {
    configurable: true,
    get: () => "complete",
  });

  runBrowserScript(dom, "assets/js/privacy-consent.js");
  await flushMicrotasks();

  const loadButton = dom.window.document
    .getElementById("comments-consent-gate")
    .querySelector('[data-comments-action="accept"]');
  loadButton.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
  await flushMicrotasks();

  const script = dom.window.document.getElementById("utterances-embed-loader");
  assert.ok(script);
  assert.equal(script.getAttribute("repo"), "hummat/hummat.github.io");
  assert.equal(script.getAttribute("issue-term"), "pathname");
  assert.equal(script.getAttribute("label"), "comments");
  assert.equal(script.getAttribute("theme"), "preferred-color-scheme");
});

test("disqus comments load only after explicit click", async () => {
  const dom = createDom({
    bodyHtml: '<div id="cookie-consent-banner" hidden></div>' + commentsFixtureHtml("disqus"),
  });
  Object.defineProperty(dom.window.document, "readyState", {
    configurable: true,
    get: () => "complete",
  });

  runBrowserScript(dom, "assets/js/privacy-consent.js");
  await flushMicrotasks();

  const loadButton = dom.window.document
    .getElementById("comments-consent-gate")
    .querySelector('[data-comments-action="accept"]');
  loadButton.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
  await flushMicrotasks();

  const script = dom.window.document.getElementById("disqus-embed-loader");
  assert.ok(script);
  assert.match(script.src, /example-shortname\.disqus\.com\/embed\.js/);
});

test("comments revoke disables embed and updates preference", async () => {
  const dom = createDom({
    bodyHtml: '<div id="cookie-consent-banner" hidden></div>' + commentsFixtureHtml("giscus"),
  });
  Object.defineProperty(dom.window.document, "readyState", {
    configurable: true,
    get: () => "complete",
  });
  dom.window.localStorage.setItem("hummat-comments-consent-v1", "accepted");

  runBrowserScript(dom, "assets/js/privacy-consent.js");
  await flushMicrotasks();

  const manage = dom.window.document.getElementById("comments-consent-manage");
  const thread = dom.window.document.getElementById("giscus_thread");
  const revokeButton = manage.querySelector('[data-comments-action="revoke"]');

  assert.equal(manage.hidden, false);
  assert.equal(thread.hidden, false);
  assert.ok(dom.window.document.getElementById("giscus-embed-loader"));

  revokeButton.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
  await flushMicrotasks();

  assert.equal(dom.window.localStorage.getItem("hummat-comments-consent-v1"), "declined");
  assert.equal(dom.window.document.getElementById("giscus-embed-loader"), null);
  assert.equal(thread.hidden, true);
  assert.equal(manage.hidden, true);
});
