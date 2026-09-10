#!/bin/sh

# This script linkifies (i.e. makes clickable in the terminal) text that appears
# to be a pull request or issue reference (e.g. #12345 or PR12345) or a
# 40-character commit hash (e.g. abc123). You can configure git to automatically
# send the output of commands that pipe their output through a pager, such as
# `git log` and `git show`, through this script by running this command from
# within your CIRCT checkout:
#
# git config core.pager 'utils/linkify.sh | pager'
#
# Consider:
#
# git config core.pager 'utils/linkify.sh | ${PAGER:-less}'
#
# The pager command is run from the root of the repository even if the git
# command is run from a subdirectory, so the relative path should always work.
#
# It requires OSC 8 support in the terminal. For a list of compatible terminals,
# see https://github.com/Alhadis/OSC8-Adoption
#
# Copied from upstream LLVM's llvm/utils/git/linkify.

sed \
  -e 's,\(#\|\bPR\)\([0-9]\+\),\x1b]8;;https://github.com/llvm/circt/issues/\2\x1b\\\0\x1b]8;;\x1b\\,gi' \
  -e 's,[0-9a-f]\{40\},\x1b]8;;https://github.com/llvm/circt/commit/\0\x1b\\\0\x1b]8;;\x1b\\,g'
