(function () {
  "use strict";

  function initPrintLinks() {
    const printLinks = document.querySelectorAll(".js-print-link");
    printLinks.forEach((link) => {
      link.addEventListener("click", (event) => {
        event.preventDefault();
        window.print();
      });
    });
  }

  function initPosteriorCalculator() {
    const trigger = document.querySelector('[data-action="calculate-posterior"]');
    if (!trigger) {
      return;
    }

    trigger.addEventListener("click", () => {
      const likelihood = Number.parseFloat(
        (document.getElementById("likelihood") || {}).value || ""
      );
      const prior = Number.parseFloat((document.getElementById("prior") || {}).value || "");
      const evidence = Number.parseFloat((document.getElementById("evidence") || {}).value || "");

      if (!Number.isFinite(likelihood) || !Number.isFinite(prior) || !Number.isFinite(evidence)) {
        return;
      }
      if (evidence === 0) {
        return;
      }

      const posterior = (likelihood * prior * 100) / evidence;
      const output = document.getElementById("posterior");
      if (output) {
        output.textContent = posterior.toFixed(2);
      }
    });
  }

  function init() {
    initPrintLinks();
    initPosteriorCalculator();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
