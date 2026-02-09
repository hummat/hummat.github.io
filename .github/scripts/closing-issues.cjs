const CLOSING_KEYWORDS = [
  'close[sd]?',
  'fix(?:e[sd])?',
  'resolve[sd]?',
  'implement[sd]?',
  'complete[sd]?',
  'address(?:e[sd])?'
].join('|');

const ISSUE_REF = '(?:#\\d+|https?:\\/\\/github\\.com\\/[^/\\s]+\\/[^/\\s]+\\/issues\\/\\d+)';
const KEYWORD_PATTERN = new RegExp(
  `\\b(?:${CLOSING_KEYWORDS})[:\\s-]*(${ISSUE_REF}(?:[,\\s]+(?:and\\s+)?${ISSUE_REF})*)`,
  'gi'
);
const SINGLE_REF_PATTERN =
  /#(\d+)|https?:\/\/github\.com\/([^/\s]+)\/([^/\s]+)\/issues\/(\d+)/gi;

function extractClosingIssueReferences(prBody, repoOwner, repoName) {
  const owner = String(repoOwner || '').toLowerCase();
  const repo = String(repoName || '').toLowerCase();
  const body = String(prBody || '');

  const issueNumbers = [];
  const skippedReferences = [];

  for (const match of body.matchAll(KEYWORD_PATTERN)) {
    const refsBlob = match[1];

    for (const refMatch of refsBlob.matchAll(SINGLE_REF_PATTERN)) {
      if (refMatch[1]) {
        issueNumbers.push(parseInt(refMatch[1], 10));
      } else {
        const refOwner = refMatch[2].toLowerCase();
        const refRepo = refMatch[3].toLowerCase();
        const issueNumber = parseInt(refMatch[4], 10);

        if (refOwner === owner && refRepo === repo) {
          issueNumbers.push(issueNumber);
        } else {
          skippedReferences.push(refMatch[0]);
        }
      }
    }
  }

  return {
    issueNumbers: [...new Set(issueNumbers)],
    skippedReferences
  };
}

module.exports = {
  extractClosingIssueReferences
};
