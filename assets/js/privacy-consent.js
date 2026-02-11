(function () {
  "use strict";

  const CONSENT_KEY = "hummat-cookie-consent-v1";
  const CONSENT_ACCEPTED = "accepted";
  const CONSENT_DECLINED = "declined";

  function readConsent() {
    try {
      return localStorage.getItem(CONSENT_KEY);
    } catch (_err) {
      return null;
    }
  }

  function writeConsent(value) {
    try {
      localStorage.setItem(CONSENT_KEY, value);
    } catch (_err) {
      // Ignore storage failures.
    }
  }

  function getAnalyticsId() {
    const meta = document.querySelector('meta[name="google-analytics-id"]');
    return meta ? (meta.getAttribute("content") || "").trim() : "";
  }

  function loadAnalytics(analyticsId) {
    if (!analyticsId || !analyticsId.startsWith("G-")) {
      return;
    }

    if (document.getElementById("ga4-loader")) {
      return;
    }

    const script = document.createElement("script");
    script.id = "ga4-loader";
    script.async = true;
    script.src = `https://www.googletagmanager.com/gtag/js?id=${encodeURIComponent(analyticsId)}`;
    document.head.appendChild(script);

    window.dataLayer = window.dataLayer || [];
    window.gtag =
      window.gtag ||
      function gtag() {
        window.dataLayer.push(arguments);
      };

    window.gtag("js", new Date());
    window.gtag("config", analyticsId);
  }

  function loadUtterances(thread) {
    if (!thread || document.getElementById("utterances-embed-loader")) {
      return;
    }

    const repo = (thread.getAttribute("data-utterances-repo") || "").trim();
    if (!repo) {
      return;
    }

    const issueTerm = (thread.getAttribute("data-utterances-issue-term") || "pathname").trim();
    const label = (thread.getAttribute("data-utterances-label") || "").trim();
    const theme = (thread.getAttribute("data-utterances-theme") || "preferred-color-scheme").trim();

    const script = document.createElement("script");
    script.id = "utterances-embed-loader";
    script.src = "https://utteranc.es/client.js";
    script.async = true;
    script.setAttribute("repo", repo);
    script.setAttribute("issue-term", issueTerm);
    if (label) {
      script.setAttribute("label", label);
    }
    script.setAttribute("theme", theme);
    script.setAttribute("crossorigin", "anonymous");

    thread.appendChild(script);
  }

  function applyOptionalServices() {
    const analyticsId = getAnalyticsId();
    const utterancesThread = document.getElementById("utterances_thread");

    loadAnalytics(analyticsId);
    loadUtterances(utterancesThread);
  }

  function initConsentBanner() {
    const banner = document.getElementById("cookie-consent-banner");
    if (!banner) {
      return;
    }

    const analyticsId = getAnalyticsId();
    const utterancesThread = document.getElementById("utterances_thread");
    const hasOptionalServices = Boolean(analyticsId || utterancesThread);

    if (!hasOptionalServices) {
      return;
    }

    const consent = readConsent();
    if (consent === CONSENT_ACCEPTED) {
      applyOptionalServices();
      return;
    }

    if (consent !== CONSENT_DECLINED) {
      banner.hidden = false;
    }

    banner.addEventListener("click", (event) => {
      const actionButton = event.target.closest("button[data-consent-action]");
      if (!actionButton) {
        return;
      }

      const action = actionButton.getAttribute("data-consent-action");
      if (action === "accept") {
        writeConsent(CONSENT_ACCEPTED);
        applyOptionalServices();
      } else if (action === "decline") {
        writeConsent(CONSENT_DECLINED);
      }

      banner.hidden = true;
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initConsentBanner);
  } else {
    initConsentBanner();
  }
})();
