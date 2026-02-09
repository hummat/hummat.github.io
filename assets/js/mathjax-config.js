window.MathJax = {
  TeX: {
    equationNumbers: {
      autoNumber: "AMS",
    },
  },
  jax: ["input/TeX", "output/CommonHTML"],
  tex2jax: {
    skipTags: ["script", "noscript", "style", "textarea", "pre"],
    inlineMath: [
      ["$", "$"],
      ["\\(", "\\)"],
    ],
    displayMath: [
      ["$$", "$$"],
      ["\\[", "\\]"],
    ],
    processEscapes: true,
  },
  CommonHTML: {
    linebreaks: {
      automatic: true,
    },
  },
  "HTML-CSS": {
    linebreaks: {
      automatic: true,
    },
  },
  SVG: {
    linebreaks: {
      automatic: true,
    },
  },
};
