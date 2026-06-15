#!/usr/bin/env python3
# Generate a Comb canonicalization scaling stress test.
#
# The produced MLIR module has N one-bit inputs and creates
# a left-associative chain of binary concats:
#        concat(concat(concat(a0, a1), a2), a3) ...
# When canonicalize flattens this, the old algorithm recreated the concat
# on each step (O(N^2)).  The stack-based rewrite is O(N).
#
# Usage:
#   python3 concat-canonicalize-scaling.py N > test.mlir
#   circt-opt -canonicalize test.mlir -o /dev/null

import sys


def generate(n: int) -> str:
  assert n >= 2, "N must be >= 2"

  out = []
  w = out.append
  w(f"// Stress test: {n} inputs, nested binary concat chain.")
  w("")

  w(f"hw.module @ConcatCanonicalizationScaling{n}(in %a0 : i1, in %a1 : i1, out z0 : i1) {{"
   )
  w("  %c0 = comb.concat %a0, %a1 : i1, i1")
  args = ", ".join(["%c0" for _ in range(n)])
  types = ", ".join(["i2" for _ in range(n)])
  w(f"  %c1 = comb.concat {args} : {types}")
  # Extract a bit from the final concat.
  w(f"  %z0 = comb.extract %c1 from {n} : (i{2 * n}) -> i1")
  w("hw.output %z0 : i1")
  w("}")
  return "\n".join(out) + "\n"


if __name__ == "__main__":
  if len(sys.argv) != 2:
    sys.stderr.write(f"usage: {sys.argv[0]} N\n")
    sys.exit(2)
  sys.stdout.write(generate(int(sys.argv[1])))
