#!/usr/bin/env bash
set -euo pipefail
CIRCT_LEC="${CIRCT_LEC:-circt-lec}"
output=$("$CIRCT_LEC" "$@" --emit-smtlib | z3 -in)
if echo "$output" | grep -q "unsat"; then
  echo "PASSED"
  exit 0
fi
echo "FAILED"
exit 1
