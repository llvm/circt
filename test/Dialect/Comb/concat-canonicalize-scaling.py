#!/usr/bin/env python3
# Generate a Comb canonicalization scaling stress test.
#
# The produced MLIR module has N one-bit inputs and creates:
#   1. A left-associative chain of binary concats:
#        concat(concat(concat(a0, a1), a2), a3) ...
#      When canonicalize flattens this, the old algorithm recreated the concat
#      on each step (O(N^2)).  The stack-based rewrite is O(N).
#   2. N extract ops that pull each individual bit back out of the final
#      N-bit concatenation result.  The old extract-of-concat logic did a
#      linear scan for every extract (O(N) per extract, O(N^2) total).
#      The batched binary-search rewrite is O(N log N).
#
# Usage:
#   python3 concat-canonicalize-scaling.py N > test.mlir
#   circt-opt -canonicalize test.mlir -o /dev/null

import sys


def generate(n: int) -> str:
  assert n >= 2, "N must be >= 2"

  out = []
  w = out.append
  w(f"// Stress test: {n} inputs, nested binary concat chain + {n} extracts.")
  w("")

  in_ports = ", ".join(f"in %a{i} : i1" for i in range(n))
  out_ports = ", ".join(f"out z{i} : i1" for i in range(n))
  w(f"hw.module @ConcatCanonicalizationScaling{n}({in_ports}, {out_ports}) {{")

  # Build a left-associative chain of binary concats.
  # Step 0: %c0 = concat(%a0, %a1)          -> i2
  # Step 1: %c1 = concat(%c0, %a2)          -> i3
  # ...
  # Step N-2: %c{N-2} = concat(%c{N-3}, %a{N-1}) -> iN
  w(f"  %c0 = comb.concat %a0, %a1 : i1, i1")
  for i in range(2, n):
    w(f"  %c{i-1} = comb.concat %c{i-2}, %a{i} : i{i}, i1")

  final = f"%c{n-2}"
  # Extract each bit from the final concat.
  extract_names = []
  for i in range(n):
    name = f"%e{i}"
    extract_names.append(name)
    w(f"  {name} = comb.extract {final} from {i} : (i{n}) -> i1")

  out_args = ", ".join(extract_names)
  out_types = ", ".join(["i1"] * n)
  w(f"  hw.output {out_args} : {out_types}")
  w("}")
  return "\n".join(out) + "\n"


if __name__ == "__main__":
  if len(sys.argv) != 2:
    sys.stderr.write(f"usage: {sys.argv[0]} N\n")
    sys.exit(2)
  sys.stdout.write(generate(int(sys.argv[1])))
