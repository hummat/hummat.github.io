const test = require("node:test");
const assert = require("node:assert/strict");
const { createDom, runBrowserScript, flushMicrotasks } = require("./test-utils");

test("consent banner loads analytics and utterances only after accept", async () => {
  const dom = createDom({
    headHtml: '<meta name="google-analytics-id" content="G-TEST1234">',
    bodyHtml:
      '<div id="cookie-consent-banner" hidden>' +
      '<button type="button" data-consent-action="accept">Accept</button>' +
      '<button type="button" data-consent-action="decline">Decline</button>' +
      "</div>" +
      '<div id="utterances_thread" data-utterances-repo="hummat/hummat.github.io" data-utterances-issue-term="pathname" data-utterances-label="comments" data-utterances-theme="preferred-color-scheme"></div>',
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
  assert.equal(dom.window.document.getElementById("utterances-embed-loader"), null);

  const acceptButton = banner.querySelector('[data-consent-action="accept"]');
  acceptButton.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
  await flushMicrotasks();

  assert.equal(dom.window.localStorage.getItem("hummat-cookie-consent-v1"), "accepted");

  const gaScript = dom.window.document.getElementById("ga4-loader");
  const utterancesScript = dom.window.document.getElementById("utterances-embed-loader");
  assert.ok(gaScript);
  assert.ok(utterancesScript);
  assert.match(gaScript.src, /googletagmanager\.com/);
  assert.match(utterancesScript.src, /utteranc\.es\/client\.js/);
  assert.equal(utterancesScript.getAttribute("repo"), "hummat/hummat.github.io");
  assert.equal(utterancesScript.getAttribute("issue-term"), "pathname");
  assert.equal(utterancesScript.getAttribute("label"), "comments");
  assert.equal(utterancesScript.getAttribute("theme"), "preferred-color-scheme");
  assert.equal(banner.hidden, true);
});
