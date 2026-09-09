#!/usr/bin/env bash
#
# Append one hls-est sweep to a long-format tracking CSV, and report what
# changed since the previous sweep.
#
# Each run is appended as a block of rows tagged with a run id, timestamp and
# git revision, so the tracking file accumulates history rather than being
# overwritten. One row per benchmark per run:
#
#   run_id,run_date,git_rev,note,directory,benchmark,status,arrays,infeasible,total_bram
#
# Usage:
#   ./track-bram-run.sh [SUMMARY_CSV] [TRACKING_CSV]
#
# Environment overrides:
#   RUN_ID   identifier for this run   (default: UTC timestamp)
#   NOTE     free-text label for what changed in this build
#   FORCE    1 = allow re-appending an existing RUN_ID

set -uo pipefail

SUMMARY="${1:-bram-results/summary.csv}"
TRACKING="${2:-bram-tracking.csv}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
NOTE="${NOTE:-}"
FORCE="${FORCE:-0}"

[[ -f "$SUMMARY" ]] || { echo "error: no such summary: $SUMMARY" >&2; exit 1; }

GIT_REV="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
  GIT_REV="$GIT_REV-dirty"
fi
RUN_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Commas in the note would corrupt the CSV.
NOTE="${NOTE//,/;}"

if [[ ! -f "$TRACKING" ]]; then
  echo "run_id,run_date,git_rev,note,directory,benchmark,status,arrays,infeasible,total_bram" > "$TRACKING"
  echo "created $TRACKING"
fi

if [[ "$FORCE" != "1" ]] && grep -q "^${RUN_ID}," "$TRACKING"; then
  echo "error: run id '$RUN_ID' is already in $TRACKING" >&2
  echo "       set RUN_ID=<something-else> or FORCE=1" >&2
  exit 1
fi

# Most recent run already recorded, used as the baseline for the delta report.
PREV_RUN="$(awk -F, 'NR>1 && $1 != "" { last = $1 } END { print last }' "$TRACKING")"

# --- append ---------------------------------------------------------------
appended=$(awk -F, -v OFS=, \
  -v run="$RUN_ID" -v date="$RUN_DATE" -v rev="$GIT_REV" -v note="$NOTE" '
  NR == 1 { next }                      # skip the summary header
  NF < 6  { next }                      # skip blank/short lines
  {
    bench = $2
    sub(/\.mlir$/, "", bench)
    print run, date, rev, note, $1, bench, $3, $4, $5, $6
    n++
  }
  END { print n > "/dev/stderr" }
' "$SUMMARY" 2>&1 >> "$TRACKING" )

echo "appended $appended row(s) as run '$RUN_ID' (rev $GIT_REV)"
[[ -n "$NOTE" ]] && echo "note: $NOTE"

# --- delta vs the previous run --------------------------------------------
if [[ -z "$PREV_RUN" ]]; then
  echo "no previous run in $TRACKING — nothing to compare against."
  exit 0
fi

echo
echo "changes vs run '$PREV_RUN':"

awk -F, -v prev="$PREV_RUN" -v cur="$RUN_ID" '
  NR == 1 { next }
  $1 == prev { pb[$6] = $10; ps[$6] = $7; seen_prev[$6] = 1 }
  $1 == cur  { cb[$6] = $10; cs[$6] = $7; seen_cur[$6]  = 1 }
  END {
    changed = 0
    for (b in seen_cur) {
      if (!(b in seen_prev)) {
        printf "  + %-34s %s (new benchmark, bram=%s)\n", b, cs[b], (cb[b] == "" ? "-" : cb[b])
        changed++
        continue
      }
      if (cs[b] != ps[b]) {
        printf "  ~ %-34s status %s -> %s\n", b, ps[b], cs[b]
        changed++
      } else if (cb[b] != pb[b]) {
        pv = (pb[b] == "" ? "-" : pb[b])
        cv = (cb[b] == "" ? "-" : cb[b])
        d  = ""
        if (pb[b] != "" && cb[b] != "") {
          diff = cb[b] - pb[b]
          d = sprintf("  (%+d)", diff)
        }
        printf "  ~ %-34s bram %s -> %s%s\n", b, pv, cv, d
        changed++
      }
    }
    for (b in seen_prev)
      if (!(b in seen_cur)) {
        printf "  - %-34s dropped from the sweep\n", b
        changed++
      }
    if (changed == 0) print "  (identical to the previous run)"

    # Totals over benchmarks both runs costed, so the comparison is like-for-like.
    for (b in seen_cur)
      if ((b in seen_prev) && pb[b] != "" && cb[b] != "") {
        tp += pb[b]; tc += cb[b]; n++
      }
    if (n > 0)
      printf "\n  %d benchmark(s) costed in both runs: total BRAM %d -> %d (%+d)\n", n, tp, tc, tc - tp
  }
' "$TRACKING"