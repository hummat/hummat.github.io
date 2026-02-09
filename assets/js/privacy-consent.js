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

  function loadDisqus(shortname) {
    if (!shortname || document.getElementById("disqus-embed-loader")) {
      return;
    }

    const dsq = document.createElement("script");
    dsq.id = "disqus-embed-loader";
    dsq.type = "text/javascript";
    dsq.async = true;
    dsq.src = `https://${shortname}.disqus.com/embed.js`;
    dsq.setAttribute("data-timestamp", String(Date.now()));

    (document.head || document.body).appendChild(dsq);
  }

  function applyOptionalServices() {
    const analyticsId = getAnalyticsId();
    const disqusThread = document.getElementById("disqus_thread");
    const disqusShortname = disqusThread
      ? (disqusThread.getAttribute("data-disqus-shortname") || "").trim()
      : "";

    loadAnalytics(analyticsId);
    loadDisqus(disqusShortname);
  }

  function initConsentBanner() {
    const banner = document.getElementById("cookie-consent-banner");
    if (!banner) {
      return;
    }

    const analyticsId = getAnalyticsId();
    const disqusThread = document.getElementById("disqus_thread");
    const hasOptionalServices = Boolean(analyticsId || disqusThread);

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
