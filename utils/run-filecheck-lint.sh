#!/usr/bin/env bash
##===- utils/run-filecheck-lint.sh - run filecheck lint ------*- Script -*-===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Runs filecheck_lint.py on FileCheck-based tests, exiting non-zero if it
# reports anything.
#
# Set FILECHECK_LINT to use a filecheck_lint.py other than the llvm submodule's.
#
##===----------------------------------------------------------------------===##

set -euo pipefail

SRC_ROOT=$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE:-$0}")")")
LINT_DIR=$SRC_ROOT/llvm/llvm/utils/filecheck_lint
FILECHECK_LINT=${FILECHECK_LINT:-$LINT_DIR/filecheck_lint.py}

cd "$SRC_ROOT"

if [ ! -f "$FILECHECK_LINT" ]; then
  echo "Unable to find $FILECHECK_LINT"
  exit 1
fi

# Run filecheck_lint.py on all files containing "RUN:" with a "FileCheck" in it.
# It always exits 0, so fail if it printed anything.
FINDINGS=$(git grep -lz "RUN:.*FileCheck" | \
  xargs -r0 -n 64 -P "$(nproc || sysctl -n hw.ncpu)" \
  python3 "$FILECHECK_LINT")

if [ -n "$FINDINGS" ]; then
  echo "$FINDINGS"
  exit 1
fi
