(function () {
  "use strict";

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

  function loadTrustedInclude($target, includeUrl) {
    $target.load(includeUrl.toString(), function (_responseText, status) {
      if (status === "error") {
        $target.text("Embedded content could not be loaded.");
      }
    });
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

      loadTrustedInclude($(this), includeUrl);
    });
  }

  function init() {
    if (!window.jQuery) {
      console.warn("jquery-enhancements.js: jQuery not loaded; features disabled.");
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
