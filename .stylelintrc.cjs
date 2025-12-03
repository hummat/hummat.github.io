module.exports = {
  customSyntax: "postcss-scss",
  extends: ["stylelint-config-recommended-scss"],
  ignoreFiles: ["style.scss", "_site/**/*", "node_modules/**/*", "figures/**/*", "data/**/*", "images/**/*"],
  rules: {
    // Relax noisy style rules for legacy theme files; keep to basic syntax checks.
    indentation: null,
    "color-hex-length": null,
    "declaration-block-single-line-max-declarations": null,
    "selector-class-pattern": null,
    "selector-pseudo-element-no-unknown": [true, { ignorePseudoElements: ["ng-deep"] }],
    "alpha-value-notation": null,
    "color-function-notation": null,
    "value-keyword-case": null,
    "scss/no-global-function-names": null,
    "scss/comment-no-empty": null,
    "at-rule-no-vendor-prefix": null,
    "property-no-vendor-prefix": null,
    "keyframes-name-pattern": null,
    "declaration-block-no-shorthand-property-overrides": null
  }
};
