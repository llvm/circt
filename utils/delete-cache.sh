#!/usr/bin/env bash

# Enhanced cache deletion script with logging
# Reads cache IDs from stdin and deletes them one by one
# Usage: ./delete-cache.sh [OPTIONS]

set -euo pipefail

# Default values
DRY_RUN=false
HELP=false

# Function to show help
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Delete GitHub Actions caches by reading cache IDs from stdin.

OPTIONS:
    -d, --dry-run    Show what would be deleted without actually deleting
    -h, --help       Show this help message

EXAMPLES:
    echo "12345" | $0
    echo -e "12345\n67890" | $0 --dry-run
    jq -r '.[].id' caches.json | $0

EOF
}

# Parse options using getopts-style manual parsing for better portability
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            HELP=true
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            show_help
            exit 1
            ;;
    esac
done

# Show help if requested
if [[ "$HELP" == "true" ]]; then
    show_help
    exit 0
fi

if [[ "$DRY_RUN" == "true" ]]; then
    echo "DRY RUN MODE: No caches will actually be deleted"
fi

count=0
while read -r id; do
  # Skip empty IDs
  if [[ -z "$id" ]]; then
    continue
  fi

  if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN] Would delete cache ID: $id"
    ((count++))
  else
    echo "Deleting cache ID: $id"
    if gh cache delete "$id"; then
      echo "Successfully deleted cache ID: $id"
      ((count++))
    else
      echo "Failed to delete cache ID: $id" >&2
    fi
    # Brief pause to avoid overwhelming the GitHub API
    # This helps prevent rate limiting issues when deleting many caches
    sleep 2
  fi
done

if [[ "$DRY_RUN" == "true" ]]; then
  echo "DRY RUN: Would have deleted $count caches"
else
  echo "Total caches deleted: $count"
fi
