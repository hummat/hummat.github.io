#!/usr/bin/env bash
set -euo pipefail

# Sync labels from labels.yml to GitHub repo
# Usage: ./sync-labels.sh [owner/repo]

REPO="${1:-hummat/hummat.github.io}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABELS_FILE="$SCRIPT_DIR/../labels.yml"

if [[ ! -f "$LABELS_FILE" ]]; then
  echo "Error: labels.yml not found at $LABELS_FILE"
  exit 1
fi

echo "Syncing labels to $REPO..."

# Parse labels.yml and create/update labels
while IFS= read -r line; do
  if [[ "$line" =~ ^-\ name:\ \"(.+)\"$ ]]; then
    name="${BASH_REMATCH[1]}"
  elif [[ "$line" =~ ^\ \ color:\ \"(.+)\"$ ]]; then
    color="${BASH_REMATCH[1]}"
  elif [[ "$line" =~ ^\ \ description:\ \"(.+)\"$ ]]; then
    description="${BASH_REMATCH[1]}"

    # Try to update existing label, create if it doesn't exist
    if gh label edit "$name" --repo "$REPO" --color "$color" --description "$description" 2>/dev/null; then
      echo "  Updated: $name"
    else
      gh label create "$name" --repo "$REPO" --color "$color" --description "$description" 2>/dev/null && \
        echo "  Created: $name" || echo "  Skipped: $name (already exists)"
    fi
  fi
done < "$LABELS_FILE"

echo "Done!"
