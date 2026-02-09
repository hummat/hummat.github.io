(function () {
  "use strict";

  function getFootnoteContent(index) {
    const id = `fn:${index}`;
    const footnote = document.getElementById(id);
    return footnote ? footnote.innerHTML.trim() : "";
  }

  function footnotePopup(showIndex, showCloseBtn) {
    const popupWrapper = document.querySelector("#popup-wrapper");
    if (!popupWrapper) {
      return;
    }

    const shouldShowIndex = showIndex !== false;
    const shouldShowCloseButton = showCloseBtn !== false;

    const popupContent = popupWrapper.appendChild(document.createElement("div"));
    popupContent.id = "popup-content";

    let popupIndex = null;
    if (shouldShowIndex) {
      popupIndex = popupWrapper.insertBefore(document.createElement("div"), popupContent);
      popupIndex.id = "popup-index";
    }

    let popupCloseButton = null;
    if (shouldShowCloseButton) {
      popupCloseButton = popupWrapper.appendChild(document.createElement("div"));
      popupCloseButton.innerHTML = "[x]";
      popupCloseButton.id = "popup-close";
    }

    const fnReturns = document.querySelectorAll("a.footnote-return");
    fnReturns.forEach((fnReturn) => {
      const parent = fnReturn.parentNode;
      if (parent) {
        parent.removeChild(fnReturn);
      }
    });

    const fnRefs = document.querySelectorAll("sup[id^='fnref:']");
    fnRefs.forEach((fnRef) => {
      fnRef.addEventListener("mouseover", createHandler("refs", fnRef));
    });

    window.addEventListener("mouseout", createHandler("close"));

    if (shouldShowCloseButton && popupCloseButton) {
      popupCloseButton.addEventListener("click", createHandler("close"));
    }

    function createHandler(type, node) {
      return function (event) {
        if (type === "close") {
          popupWrapper.style.display = "none";
          return;
        }

        if (!node || !event) {
          return;
        }

        event.preventDefault();
        const index = node.id.substring(6);

        if (shouldShowIndex && popupIndex) {
          popupIndex.innerHTML = `${index}.`;
        }

        popupContent.innerHTML = getFootnoteContent(index);
        popupWrapper.style.display = "flex";
      };
    }
  }

  function init() {
    footnotePopup(true, false);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
