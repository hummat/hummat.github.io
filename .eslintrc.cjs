module.exports = {
  env: {
    browser: true,
    es2021: true
  },
  plugins: ["html"],
  extends: ["eslint:recommended"],
  parserOptions: {
    ecmaVersion: "latest",
    sourceType: "script"
  },
  globals: {
    $: "readonly",
    jQuery: "readonly",
    MathJax: "readonly",
    Plotly: "readonly"
  },
  rules: {
    "no-unused-vars": ["warn", { args: "none" }],
    "no-console": "off",
    "prefer-const": "warn"
  }
};
