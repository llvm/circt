#!/usr/bin/env bash

# Cache cleanup script for CI workflows
# This script identifies and deletes cache entries based on size and age criteria
# to maintain cache efficiency and remove caches from failed builds.

set -euo pipefail

# Default values
DRY_RUN=false
CACHE_KEY_PATTERN=""
CACHE_REF=""
HELP=false

# Function to show help
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Cache cleanup script that removes small caches (likely from failed builds)
and maintains only the most recent successful builds.

OPTIONS:
    -p, --pattern PATTERN    Cache key pattern to match (required)
    -r, --ref REF           Git reference for cache cleanup (default: current branch)
    -d, --dry-run           Show what would be deleted without actually deleting
    -h, --help              Show this help message

EXAMPLES:
    $0 -p "ccache-linux" -r "refs/heads/main"
    $0 --pattern "build-cache" --dry-run
    $0 -p "test-cache" -r "\${{ github.ref }}"

DESCRIPTION:
    This script uses a dynamic size threshold of 1/2 of maximum cache size
    as the cleanup threshold. It employs regex pattern matching to find caches
    with '.*PATTERN.*' for flexible matching. The system keeps the top 1 cache
    to preserve the most recent successful build.

    The cache cleanup logic:
    1. Deletes all caches smaller than the threshold (failed builds)
    2. Deletes caches beyond top 1 that exceed the threshold (old builds)

EOF
}

# Parse command line arguments using getopts
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--pattern)
            CACHE_KEY_PATTERN="$2"
            shift 2
            ;;
        -r|--ref)
            CACHE_REF="$2"
            shift 2
            ;;
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

# Validate required parameters
if [[ -z "$CACHE_KEY_PATTERN" ]]; then
    echo "ERROR: cache key pattern is required" >&2
    echo "Use -p/--pattern to specify the cache key pattern" >&2
    show_help
    exit 1
fi

# Use current branch if ref not specified
if [[ -z "$CACHE_REF" ]]; then
    CACHE_REF=$(git rev-parse --abbrev-ref HEAD)
    echo "No ref specified, using current branch: $CACHE_REF"
fi

echo "=== Cache Cleanup Configuration ==="
echo "Pattern: .*${CACHE_KEY_PATTERN}.*"
echo "Reference: $CACHE_REF"
echo "Dry run: $DRY_RUN"
echo "=================================="

# Get all caches and filter with regex pattern
echo "Fetching all caches and filtering with regex pattern..."
ALL_CACHES=$(gh cache list -L 200 --json id,key,sizeInBytes)
MATCHING_CACHES=$(echo "$ALL_CACHES" | jq --arg pattern ".*${CACHE_KEY_PATTERN}.*" '[.[] | select(.key | test($pattern))]')

echo "Found $(echo "$MATCHING_CACHES" | jq 'length') matching caches"
echo "Matching caches:"
echo "$MATCHING_CACHES" | jq -r '.[] | "  \(.key) - \(.sizeInBytes) bytes"'

# Calculate the size threshold as 1/2 of the maximum cache size
MAX_SIZE=$(echo "$MATCHING_CACHES" | jq 'map(.sizeInBytes) | max // 0')
SIZE_THRESHOLD=$((MAX_SIZE / 2))

echo ""
echo "=== Cleanup Analysis ==="
echo "Maximum cache size: $MAX_SIZE bytes"
echo "Size threshold (1/2 of max): $SIZE_THRESHOLD bytes"
echo "Will delete:"
echo "  - All caches smaller than $SIZE_THRESHOLD bytes (likely failed builds)"
echo "  - Caches beyond the top 1 (if they are >= $SIZE_THRESHOLD bytes)"
echo "========================"

# Delete only the caches in the current branch
echo ""
echo "Fetching caches for reference: $CACHE_REF"
MATCHING_CACHES_IN_CURRENT_BRANCH=$(gh cache list -L 200 --json id,key,sizeInBytes --ref "$CACHE_REF" | jq --arg pattern ".*${CACHE_KEY_PATTERN}.*" '[.[] | select(.key | test($pattern))]')

# Get cache info with size and filter:
# 1. All caches smaller than threshold (failed builds)
# 2. Caches beyond top 1 that are >= threshold (old successful builds)
if [[ "$MAX_SIZE" -gt 0 ]]; then
    echo "Processing caches for deletion..."
    
    # Build the deletion command arguments
    DELETE_ARGS=""
    if [[ "$DRY_RUN" == "true" ]]; then
        DELETE_ARGS="--dry-run"
    fi
    
    # Pipe the cache IDs to the delete script
    echo "$MATCHING_CACHES_IN_CURRENT_BRANCH" | \
        jq --argjson threshold "$SIZE_THRESHOLD" \
        '[.[] | select(.sizeInBytes < $threshold)] + (.[1:] | map(select(.sizeInBytes >= $threshold))) | .[].id' | \
        "$(dirname "$0")/delete-cache.sh" $DELETE_ARGS
else
    echo "No caches found matching pattern: .*${CACHE_KEY_PATTERN}.*"
fi

echo ""
echo "Cache cleanup completed."
