#!/usr/bin/env bash
set -euo pipefail

nix develop /home/roland/circt-nix#circt \
  -c bash -lc "ninja -C build -j${JOBS:-3} bin/circt-opt"
