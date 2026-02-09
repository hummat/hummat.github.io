const fs = require("node:fs");
const path = require("node:path");
const { JSDOM } = require("jsdom");

function createDom({ bodyHtml = "", headHtml = "", url = "https://hummat.com/" } = {}) {
  return new JSDOM(`<!doctype html><html><head>${headHtml}</head><body>${bodyHtml}</body></html>`, {
    url,
    runScripts: "outside-only",
    pretendToBeVisual: true,
  });
}

function runBrowserScript(dom, relativeScriptPath) {
  const script = fs.readFileSync(path.join(process.cwd(), relativeScriptPath), "utf8");
  dom.window.eval(script);
}

function triggerDomContentLoaded(dom) {
  dom.window.document.dispatchEvent(
    new dom.window.Event("DOMContentLoaded", {
      bubbles: true,
      cancelable: true,
    })
  );
}

async function flushMicrotasks(turns = 3) {
  for (let i = 0; i < turns; i += 1) {
    // eslint-disable-next-line no-await-in-loop
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
}

module.exports = {
  createDom,
  runBrowserScript,
  triggerDomContentLoaded,
  flushMicrotasks,
};
