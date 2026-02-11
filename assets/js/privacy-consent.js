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

  function resolveCommentsThread() {
    return (
      document.getElementById("giscus_thread") ||
      document.getElementById("utterances_thread") ||
      document.getElementById("disqus_thread")
    );
  }

  function resolveCommentsProvider(thread) {
    if (!thread) {
      return "";
    }

    if (thread.id === "giscus_thread") {
      return "giscus";
    }
    if (thread.id === "utterances_thread") {
      return "utterances";
    }
    if (thread.id === "disqus_thread") {
      return "disqus";
    }

    const containerProvider = thread.closest(".comments")?.getAttribute("data-comments-provider");
    return containerProvider || "";
  }

  function removeGiscus(thread) {
    if (!thread) {
      return;
    }

    const embeddedElements = thread.querySelectorAll(
      "script#giscus-embed-loader, iframe.giscus-frame"
    );
    embeddedElements.forEach((element) => {
      element.remove();
    });

    const detachedFrames = document.querySelectorAll("iframe.giscus-frame");
    detachedFrames.forEach((frame) => {
      frame.remove();
    });

    thread.hidden = true;
  }

  function loadGiscus(thread) {
    if (!thread || document.getElementById("giscus-embed-loader")) {
      return;
    }

    const repo = (thread.getAttribute("data-giscus-repo") || "").trim();
    const repoId = (thread.getAttribute("data-giscus-repo-id") || "").trim();
    const category = (thread.getAttribute("data-giscus-category") || "").trim();
    const categoryId = (thread.getAttribute("data-giscus-category-id") || "").trim();
    if (!repo || !repoId || !category || !categoryId) {
      return;
    }

    const mapping = (thread.getAttribute("data-giscus-mapping") || "pathname").trim();
    const strict = (thread.getAttribute("data-giscus-strict") || "0").trim();
    const reactionsEnabled = (thread.getAttribute("data-giscus-reactions-enabled") || "1").trim();
    const emitMetadata = (thread.getAttribute("data-giscus-emit-metadata") || "0").trim();
    const inputPosition = (thread.getAttribute("data-giscus-input-position") || "top").trim();
    const theme = (thread.getAttribute("data-giscus-theme") || "preferred_color_scheme").trim();
    const lang = (thread.getAttribute("data-giscus-lang") || "en").trim();

    const script = document.createElement("script");
    script.id = "giscus-embed-loader";
    script.src = "https://giscus.app/client.js";
    script.async = true;
    script.setAttribute("data-repo", repo);
    script.setAttribute("data-repo-id", repoId);
    script.setAttribute("data-category", category);
    script.setAttribute("data-category-id", categoryId);
    script.setAttribute("data-mapping", mapping);
    script.setAttribute("data-strict", strict);
    script.setAttribute("data-reactions-enabled", reactionsEnabled);
    script.setAttribute("data-emit-metadata", emitMetadata);
    script.setAttribute("data-input-position", inputPosition);
    script.setAttribute("data-theme", theme);
    script.setAttribute("data-lang", lang);
    script.setAttribute("crossorigin", "anonymous");

    thread.hidden = false;
    thread.appendChild(script);
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

  function removeDisqus(thread) {
    if (!thread) {
      return;
    }

    const embeddedElements = thread.querySelectorAll("script#disqus-embed-loader");
    embeddedElements.forEach((element) => {
      element.remove();
    });

    const detachedFrames = document.querySelectorAll("iframe[src*='disqus.com']");
    detachedFrames.forEach((frame) => {
      frame.remove();
    });

    const detachedScripts = document.querySelectorAll("script[src*='disqus.com']");
    detachedScripts.forEach((script) => {
      if (script.id !== "disqus-embed-loader") {
        script.remove();
      }
    });

    const comments = thread.querySelectorAll("*");
    comments.forEach((element) => {
      element.remove();
    });

    thread.hidden = true;
  }

  function loadDisqus(thread) {
    if (!thread || document.getElementById("disqus-embed-loader")) {
      return;
    }

    const shortname = (thread.getAttribute("data-disqus-shortname") || "").trim();
    if (!shortname) {
      return;
    }

    const script = document.createElement("script");
    script.id = "disqus-embed-loader";
    script.src = `https://${shortname}.disqus.com/embed.js`;
    script.async = true;
    script.setAttribute("data-timestamp", String(Date.now()));

    thread.hidden = false;
    thread.appendChild(script);
  }

  function removeProviderEmbed(provider, thread) {
    if (provider === "giscus") {
      removeGiscus(thread);
      return;
    }
    if (provider === "utterances") {
      removeUtterances(thread);
      return;
    }
    if (provider === "disqus") {
      removeDisqus(thread);
    }
  }

  function loadProviderEmbed(provider, thread) {
    if (provider === "giscus") {
      loadGiscus(thread);
      return;
    }
    if (provider === "utterances") {
      loadUtterances(thread);
      return;
    }
    if (provider === "disqus") {
      loadDisqus(thread);
    }
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
    const thread = resolveCommentsThread();
    if (!thread) {
      return;
    }

    const provider = resolveCommentsProvider(thread);
    if (!provider) {
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
      loadProviderEmbed(provider, thread);
    } else {
      removeProviderEmbed(provider, thread);
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
        loadProviderEmbed(provider, thread);
      } else if (action === "revoke") {
        writePreference(COMMENTS_CONSENT_KEY, CONSENT_DECLINED);
        removeProviderEmbed(provider, thread);
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
