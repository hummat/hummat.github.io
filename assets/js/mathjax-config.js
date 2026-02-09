window.MathJax = {
  tex: {
    inlineMath: [
      ["$", "$"],
      ["\\(", "\\)"],
    ],
    displayMath: [
      ["$$", "$$"],
      ["\\[", "\\]"],
    ],
    processEscapes: true,
    processEnvironments: true,
    tags: "ams",
  },
  chtml: {
    matchFontHeight: true,
  },
  output: {
    displayOverflow: "linebreak",
    linebreaks: {
      inline: true,
      width: "100%",
    },
  },
};
