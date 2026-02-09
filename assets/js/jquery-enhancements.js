(function () {
  "use strict";

  const DEFAULT_INCLUDE_HEIGHT = 560;
  const TRUSTED_INCLUDE_HOSTS = new Set(["assets.hummat.com", window.location.hostname]);

  function toggleGif($img) {
    const src = $img.attr("src") || "";
    if (src.endsWith(".png")) {
      $img.attr("src", src.replace(".png", ".gif"));
    } else {
      $img.attr("src", src.replace(".gif", ".png"));
    }
  }

  function startGif($img) {
    const src = $img.attr("src") || "";
    $img.attr("src", src.replace(".png", ".gif"));
  }

  function stopGif($img) {
    const src = $img.attr("src") || "";
    $img.attr("src", src.replace(".gif", ".png"));
  }

  function parseIncludeHeight(element) {
    const raw = element.getAttribute("data-include-height");
    const parsed = Number.parseInt(raw || "", 10);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_INCLUDE_HEIGHT;
  }

  function getSafeIncludeUrl(raw) {
    if (!raw) {
      return null;
    }

    let parsed;
    try {
      parsed = new URL(raw, window.location.origin);
    } catch (_err) {
      return null;
    }

    const isLocalDev =
      window.location.hostname === "127.0.0.1" || window.location.hostname === "localhost";
    const isLocalSameOriginHttp = isLocalDev && parsed.origin === window.location.origin;
    const hasSafeProtocol = parsed.protocol === "https:" || isLocalSameOriginHttp;

    if (!hasSafeProtocol) {
      return null;
    }

    if (!TRUSTED_INCLUDE_HOSTS.has(parsed.hostname)) {
      return null;
    }

    return parsed;
  }

  function replaceIncludeWithIframe(element, includeUrl) {
    const frame = document.createElement("iframe");
    const fallbackClassName = element.className ? ` ${element.className}` : "";
    const height = parseIncludeHeight(element);

    frame.className = `embedded-include${fallbackClassName}`;
    frame.src = includeUrl.toString();
    frame.loading = "lazy";
    frame.referrerPolicy = "no-referrer";
    frame.sandbox = "allow-scripts allow-popups";
    frame.title = element.getAttribute("data-include-title") || "Embedded interactive content";
    frame.style.width = "100%";
    frame.style.minHeight = `${height}px`;
    frame.style.border = "0";

    const inlineStyle = element.getAttribute("style");
    if (inlineStyle) {
      frame.setAttribute("style", `${inlineStyle}; min-height:${height}px; border:0;`);
    }

    element.replaceWith(frame);
  }

  function initAnimatedImages($) {
    $(".img-animate").each(function () {
      $(this).on("mouseenter", function () {
        startGif($(this));
      });
      $(this).on("mouseleave", function () {
        stopGif($(this));
      });
      $(this).on("touchstart", function () {
        toggleGif($(this));
      });
    });
  }

  function initSwapImages($) {
    $(".img-swap").each(function () {
      const src = $(this).attr("src") || "";
      $(this).hover(
        function () {
          $(this).attr("src", src.replace("jpg", "png"));
        },
        function () {
          $(this).attr("src", src.replace("png", "jpg"));
        }
      );
    });
  }

  function initDataIncludes($) {
    const includes = $("[data-include]");

    $.each(includes, function () {
      const includeUrl = getSafeIncludeUrl($(this).data("include"));
      if (!includeUrl) {
        this.textContent = "Embedded content blocked: untrusted source.";
        return;
      }

      replaceIncludeWithIframe(this, includeUrl);
    });
  }

  function init() {
    if (!window.jQuery) {
      return;
    }

    const $ = window.jQuery;
    initAnimatedImages($);
    initSwapImages($);
    initDataIncludes($);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
