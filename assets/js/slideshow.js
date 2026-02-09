(function () {
  "use strict";

  let slideIndex = 1;

  function showSlides(n) {
    const slides = document.getElementsByClassName("mySlides");
    const dots = document.getElementsByClassName("dot");
    if (slides.length === 0) {
      return;
    }

    if (n > slides.length) {
      slideIndex = 1;
    }
    if (n < 1) {
      slideIndex = slides.length;
    }

    for (let i = 0; i < slides.length; i += 1) {
      slides[i].style.display = "none";
      slides[i].setAttribute("aria-hidden", "true");
    }

    for (let i = 0; i < dots.length; i += 1) {
      dots[i].className = dots[i].className.replace(" active", "");
      dots[i].setAttribute("aria-selected", "false");
    }

    slides[slideIndex - 1].style.display = "block";
    slides[slideIndex - 1].setAttribute("aria-hidden", "false");

    if (dots.length > 0 && dots[slideIndex - 1]) {
      dots[slideIndex - 1].className += " active";
      dots[slideIndex - 1].setAttribute("aria-selected", "true");
    }

    const liveRegion = document.getElementById("slideshow-live-region");
    if (liveRegion) {
      liveRegion.textContent = `Slide ${slideIndex} of ${slides.length}`;
    }
  }

  function getStepFromControl(control) {
    const explicitStep = Number.parseInt(control.getAttribute("data-slide-step") || "", 10);
    if (Number.isFinite(explicitStep) && explicitStep !== 0) {
      return explicitStep;
    }
    return control.classList.contains("prev") ? -1 : 1;
  }

  function getSlideNumberFromDot(dot) {
    const explicitIndex = Number.parseInt(dot.getAttribute("data-slide-to") || "", 10);
    if (Number.isFinite(explicitIndex) && explicitIndex > 0) {
      return explicitIndex;
    }

    const dots = Array.from(document.getElementsByClassName("dot"));
    return dots.indexOf(dot) + 1;
  }

  function plusSlides(step) {
    showSlides((slideIndex += step));
  }

  function currentSlide(index) {
    showSlides((slideIndex = index));
  }

  function init() {
    showSlides(slideIndex);

    document.addEventListener("click", (event) => {
      const control = event.target.closest(".prev, .next");
      if (control) {
        event.preventDefault();
        plusSlides(getStepFromControl(control));
        return;
      }

      const dot = event.target.closest(".dot");
      if (dot) {
        event.preventDefault();
        currentSlide(getSlideNumberFromDot(dot));
      }
    });

    const container = document.querySelector(".slideshow-container");
    if (container) {
      container.setAttribute("tabindex", "0");
      container.setAttribute("role", "region");
      container.setAttribute("aria-label", "Image slideshow");

      const liveRegion = document.createElement("div");
      liveRegion.id = "slideshow-live-region";
      liveRegion.setAttribute("aria-live", "polite");
      liveRegion.setAttribute("aria-atomic", "true");
      liveRegion.className = "sr-only";
      container.appendChild(liveRegion);

      container.addEventListener("keydown", (event) => {
        if (event.key === "ArrowLeft") {
          event.preventDefault();
          plusSlides(-1);
        } else if (event.key === "ArrowRight") {
          event.preventDefault();
          plusSlides(1);
        }
      });
    }

    const prevButtons = document.querySelectorAll(".prev");
    prevButtons.forEach((button) => {
      button.setAttribute("role", "button");
      button.setAttribute("aria-label", "Previous slide");
    });

    const nextButtons = document.querySelectorAll(".next");
    nextButtons.forEach((button) => {
      button.setAttribute("role", "button");
      button.setAttribute("aria-label", "Next slide");
    });

    const dots = document.getElementsByClassName("dot");
    for (let i = 0; i < dots.length; i += 1) {
      dots[i].setAttribute("role", "button");
      dots[i].setAttribute("aria-label", `Go to slide ${i + 1}`);
    }
  }

  window.plusSlides = plusSlides;
  window.currentSlide = currentSlide;

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
