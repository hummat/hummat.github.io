module.exports = {
  env: {
    browser: true,
    es2021: true,
  },
  plugins: ["html"],
  extends: ["eslint:recommended"],
  parserOptions: {
    ecmaVersion: "latest",
    sourceType: "script",
  },
  globals: {
    $: "readonly",
    jQuery: "readonly",
    MathJax: "readonly",
    Plotly: "readonly",
  },
  rules: {
    eqeqeq: ["error", "always", { null: "ignore" }],
    "no-unused-vars": ["error", { args: "none" }],
    "no-console": "off",
    "prefer-const": "error",
  },
  overrides: [
    {
      files: ["tests/**/*.js"],
      env: {
        browser: false,
        es2021: true,
        node: true,
      },
      globals: {
        $: "off",
        jQuery: "off",
        MathJax: "off",
        Plotly: "off",
      },
    },
  ],
};
