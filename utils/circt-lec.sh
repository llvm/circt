#!/usr/bin/env bash
# CI-only test wrapper around circt-lec that uses z3 to check equivalence.
# This avoids slow JIT execution of circt-lec in Debug builds.
set -euo pipefail
CIRCT_LEC="${CIRCT_LEC:-circt-lec}"
output=$("$CIRCT_LEC" "$@" --emit-smtlib | z3 -in)
if echo "$output" | grep -q "unsat"; then
  echo "PASSED"
  exit 0
fi
echo "FAILED"
exit 1
