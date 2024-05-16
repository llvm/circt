#!/usr/bin/env bash
##===- utils/run-filecheck-lint.sh - run filecheck lint ------*- Script -*-===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Runs filecheck_lint.py on FileCheck-based tests.
#
##===----------------------------------------------------------------------===##

set -euo pipefail

SRC_ROOT=$(dirname $(dirname $(readlink -f "${BASH_SOURCE:-$0}")))

cd "$SRC_ROOT"

if [ -x "$(command -v python3)" ]; then
  PYTHON=python3
elif [ -x "$(command -v python2)" ]; then
  PYTHON=python2
elif [ -x "$(command -v python)" ]; then
  PYTHON=python
else
  echo "Unable to find python"
  exit 1
fi

# Run filecheck_lint.py on all files cotaining "RUN:" with a "FileCheck" in it.
git grep -lz "RUN:.*FileCheck" | \
  xargs -r0 \
  "${PYTHON}" "${SRC_ROOT}/llvm/llvm/utils/filecheck_lint/filecheck_lint.py"
