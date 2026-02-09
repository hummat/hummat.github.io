(function () {
  "use strict";

  function showSlides(n, slideshow) {
    const slides = slideshow.getElementsByClassName("mySlides");
    const dots = slideshow.getElementsByClassName("dot");
    if (slides.length === 0) {
      return;
    }

    if (n > slides.length) {
      slideshow.currentSlideIndex = 1;
    }
    if (n < 1) {
      slideshow.currentSlideIndex = slides.length;
    }

    for (let i = 0; i < slides.length; i += 1) {
      slides[i].style.display = "none";
      slides[i].setAttribute("aria-hidden", "true");
    }

    for (let i = 0; i < dots.length; i += 1) {
      dots[i].className = dots[i].className.replace(" active", "");
      dots[i].setAttribute("aria-selected", "false");
    }

    slides[slideshow.currentSlideIndex - 1].style.display = "block";
    slides[slideshow.currentSlideIndex - 1].setAttribute("aria-hidden", "false");

    if (dots.length > 0 && dots[slideshow.currentSlideIndex - 1]) {
      dots[slideshow.currentSlideIndex - 1].className += " active";
      dots[slideshow.currentSlideIndex - 1].setAttribute("aria-selected", "true");
    }

    const liveRegion = slideshow.querySelector(".slideshow-live-region");
    if (liveRegion) {
      liveRegion.textContent = `Slide ${slideshow.currentSlideIndex} of ${slides.length}`;
    }
  }

  function plusSlides(step, slideshow) {
    if (!slideshow) {
      return;
    }
    showSlides((slideshow.currentSlideIndex += step), slideshow);
  }

  function currentSlide(index, slideshow) {
    if (!slideshow) {
      return;
    }
    showSlides((slideshow.currentSlideIndex = index), slideshow);
  }

  function stepFromControl(control) {
    const explicitStep = Number.parseInt(control.getAttribute("data-slide-step") || "", 10);
    if (Number.isFinite(explicitStep) && explicitStep !== 0) {
      return explicitStep;
    }
    return control.classList.contains("prev") ? -1 : 1;
  }

  function slideFromDot(dot, slideshow) {
    const explicitIndex = Number.parseInt(dot.getAttribute("data-slide-to") || "", 10);
    if (Number.isFinite(explicitIndex) && explicitIndex > 0) {
      return explicitIndex;
    }

    const dots = Array.from(slideshow.getElementsByClassName("dot"));
    return dots.indexOf(dot) + 1;
  }

  function initSlideshow(slideshow, index) {
    slideshow.currentSlideIndex = 1;
    showSlides(slideshow.currentSlideIndex, slideshow);

    slideshow.setAttribute("tabindex", "0");
    slideshow.setAttribute("role", "region");
    slideshow.setAttribute("aria-label", `Image slideshow ${index + 1}`);

    const liveRegion = document.createElement("div");
    liveRegion.className = "slideshow-live-region sr-only";
    liveRegion.setAttribute("aria-live", "polite");
    liveRegion.setAttribute("aria-atomic", "true");
    slideshow.appendChild(liveRegion);

    slideshow.addEventListener("keydown", (event) => {
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        plusSlides(-1, slideshow);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        plusSlides(1, slideshow);
      }
    });

    slideshow.addEventListener("click", (event) => {
      const control = event.target.closest(".prev, .next");
      if (control) {
        event.preventDefault();
        plusSlides(stepFromControl(control), slideshow);
        return;
      }

      const dot = event.target.closest(".dot");
      if (dot) {
        event.preventDefault();
        currentSlide(slideFromDot(dot, slideshow), slideshow);
      }
    });

    const prevButtons = slideshow.querySelectorAll(".prev");
    prevButtons.forEach((button) => {
      button.setAttribute("role", "button");
      button.setAttribute("aria-label", "Previous slide");
    });

    const nextButtons = slideshow.querySelectorAll(".next");
    nextButtons.forEach((button) => {
      button.setAttribute("role", "button");
      button.setAttribute("aria-label", "Next slide");
    });

    const dots = slideshow.getElementsByClassName("dot");
    for (let i = 0; i < dots.length; i += 1) {
      dots[i].setAttribute("role", "button");
      dots[i].setAttribute("aria-label", `Go to slide ${i + 1}`);
    }
  }

  function init() {
    const slideshows = document.querySelectorAll('[id^="slideshow"]');
    slideshows.forEach((slideshow, index) => {
      initSlideshow(slideshow, index);
    });
  }

  window.plusSlides = plusSlides;
  window.currentSlide = currentSlide;

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
