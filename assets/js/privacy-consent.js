(function () {
  "use strict";

  const CONSENT_KEY = "hummat-cookie-consent-v1";
  const COMMENTS_CONSENT_KEY = "hummat-comments-consent-v1";
  const CONSENT_ACCEPTED = "accepted";
  const CONSENT_DECLINED = "declined";

  function readPreference(key) {
    try {
      return localStorage.getItem(key);
    } catch (_err) {
      return null;
    }
  }

  function writePreference(key, value) {
    try {
      localStorage.setItem(key, value);
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

  function removeUtterances(thread) {
    if (!thread) {
      return;
    }

    const embeddedElements = thread.querySelectorAll(
      "script#utterances-embed-loader, iframe.utterances-frame"
    );
    embeddedElements.forEach((element) => {
      element.remove();
    });

    const detachedFrames = document.querySelectorAll("iframe.utterances-frame");
    detachedFrames.forEach((frame) => {
      frame.remove();
    });

    thread.hidden = true;
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

    thread.hidden = false;
    thread.appendChild(script);
  }

  function setCommentsUiState({ loaded, gate, manage, thread }) {
    if (gate) {
      gate.hidden = loaded;
    }

    if (manage) {
      manage.hidden = !loaded;
    }

    if (thread) {
      thread.hidden = !loaded;
    }
  }

  function initCommentsConsent() {
    const thread = document.getElementById("utterances_thread");
    if (!thread) {
      return;
    }

    const commentsRoot = thread.closest(".comments");
    const gate = document.getElementById("comments-consent-gate");
    const manage = document.getElementById("comments-consent-manage");
    if (!commentsRoot || !gate) {
      return;
    }

    const commentsConsent = readPreference(COMMENTS_CONSENT_KEY);
    const shouldLoadComments = commentsConsent === CONSENT_ACCEPTED;

    setCommentsUiState({
      loaded: shouldLoadComments,
      gate,
      manage,
      thread,
    });

    if (shouldLoadComments) {
      loadUtterances(thread);
    } else {
      removeUtterances(thread);
    }

    commentsRoot.addEventListener("click", (event) => {
      const actionButton = event.target.closest("button[data-comments-action]");
      if (!actionButton) {
        return;
      }

      const action = actionButton.getAttribute("data-comments-action");
      if (action === "accept") {
        writePreference(COMMENTS_CONSENT_KEY, CONSENT_ACCEPTED);
        setCommentsUiState({
          loaded: true,
          gate,
          manage,
          thread,
        });
        loadUtterances(thread);
      } else if (action === "revoke") {
        writePreference(COMMENTS_CONSENT_KEY, CONSENT_DECLINED);
        removeUtterances(thread);
        setCommentsUiState({
          loaded: false,
          gate,
          manage,
          thread,
        });
      }
    });
  }

  function initConsentBanner() {
    const banner = document.getElementById("cookie-consent-banner");
    if (!banner) {
      return;
    }

    const analyticsId = getAnalyticsId();
    const hasAnalytics = Boolean(analyticsId);

    if (!hasAnalytics) {
      return;
    }

    const consent = readPreference(CONSENT_KEY);
    if (consent === CONSENT_ACCEPTED) {
      loadAnalytics(analyticsId);
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
        writePreference(CONSENT_KEY, CONSENT_ACCEPTED);
        loadAnalytics(analyticsId);
      } else if (action === "decline") {
        writePreference(CONSENT_KEY, CONSENT_DECLINED);
      }

      banner.hidden = true;
    });
  }

  function initPrivacyControls() {
    initConsentBanner();
    initCommentsConsent();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initPrivacyControls);
  } else {
    initPrivacyControls();
  }
})();
