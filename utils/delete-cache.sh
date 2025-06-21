#!/usr/bin/env bash

# Enhanced cache deletion script with logging
# Reads cache IDs from stdin and deletes them
# Usage: ./delete-cache.sh [--dry-run]

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
  DRY_RUN=true
  echo "DRY RUN MODE: No caches will actually be deleted"
fi

count=0
while read -r id; do
  if [[ -n "$id" ]]; then
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
      sleep 2
    fi
  fi
done

if [[ "$DRY_RUN" == "true" ]]; then
  echo "DRY RUN: Would have deleted $count caches"
else
  echo "Total caches deleted: $count"
fi
