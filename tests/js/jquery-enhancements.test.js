const test = require("node:test");
const assert = require("node:assert/strict");
const jqueryFactory = require("jquery");
const { createDom, runBrowserScript, flushMicrotasks } = require("./test-utils");

test("jquery enhancements warns when jQuery is unavailable", async () => {
  const dom = createDom({
    bodyHtml:
      '<div id="target" data-include="https://assets.hummat.com/figures/example.html"></div>',
  });
  const warnings = [];
  dom.window.console.warn = (message) => warnings.push(message);
  Object.defineProperty(dom.window.document, "readyState", {
    configurable: true,
    get: () => "complete",
  });

  runBrowserScript(dom, "assets/js/jquery-enhancements.js");
  await flushMicrotasks();

  assert.equal(warnings.length, 1);
  assert.match(warnings[0], /jQuery not loaded/i);
});

test("jquery enhancements loads trusted includes and blocks untrusted ones", async () => {
  const dom = createDom({
    bodyHtml:
      '<div id="trusted" data-include="https://assets.hummat.com/figures/example.html"></div>' +
      '<div id="blocked" data-include="https://evil.example/figure.html"></div>',
  });

  const $ = jqueryFactory(dom.window);
  dom.window.$ = $;
  dom.window.jQuery = $;
  Object.defineProperty(dom.window.document, "readyState", {
    configurable: true,
    get: () => "complete",
  });

  const loadCalls = [];
  $.fn.load = function load(url, complete) {
    loadCalls.push({ id: this.attr("id"), url });
    if (typeof complete === "function") {
      complete.call(this[0], "", "success");
    }
    return this;
  };

  runBrowserScript(dom, "assets/js/jquery-enhancements.js");
  await flushMicrotasks();

  assert.equal(loadCalls.length, 1);
  assert.equal(loadCalls[0].id, "trusted");
  assert.equal(loadCalls[0].url, "https://assets.hummat.com/figures/example.html");
  assert.match(dom.window.document.getElementById("blocked").textContent, /blocked/i);
});
