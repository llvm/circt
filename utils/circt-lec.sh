#!/usr/bin/env bash
set -euo pipefail
CIRCT_LEC="${CIRCT_LEC:-circt-lec}"
"$CIRCT_LEC" "$@" --emit-smtlib | z3 -in | grep -q "unsat"
