const test = require('node:test');
const assert = require('node:assert/strict');

const { extractClosingIssueReferences } = require('./closing-issues.cjs');

test('extracts all shorthand refs in a multi-reference sentence', () => {
  const body = 'Closes #1, #2, #3 and #4';
  const result = extractClosingIssueReferences(body, 'hummat', 'hummat.github.io');

  assert.deepEqual(result.issueNumbers, [1, 2, 3, 4]);
  assert.deepEqual(result.skippedReferences, []);
});

test('extracts mixed shorthand and same-repo URL refs', () => {
  const body =
    'Fixes #10, https://github.com/hummat/hummat.github.io/issues/20 and #30';
  const result = extractClosingIssueReferences(body, 'hummat', 'hummat.github.io');

  assert.deepEqual(result.issueNumbers, [10, 20, 30]);
  assert.deepEqual(result.skippedReferences, []);
});

test('ignores cross-repo URL refs', () => {
  const body =
    'Resolves https://github.com/other/repo/issues/5, #6 and ' +
    'https://github.com/hummat/hummat.github.io/issues/7';
  const result = extractClosingIssueReferences(body, 'hummat', 'hummat.github.io');

  assert.deepEqual(result.issueNumbers, [6, 7]);
  assert.deepEqual(result.skippedReferences, ['https://github.com/other/repo/issues/5']);
});

test('deduplicates refs across clauses', () => {
  const body = 'Fixes #11. Also closes #11 and #12.';
  const result = extractClosingIssueReferences(body, 'hummat', 'hummat.github.io');

  assert.deepEqual(result.issueNumbers, [11, 12]);
});
