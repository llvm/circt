#!/usr/bin/env python3
# Generate a HW array_get with constant indices scaling stress test.
#
# The produced MLIR module has an array with N elements and N
# array_get ops, one per element using a constant index.
#
# Usage:
#   python3 constant-array-gets.py N > test.mlir
#   circt-opt -hw-aggregate-to-comb test.mlir -o /dev/null

import sys


def generate(n: int) -> str:
  assert n >= 1, "N must be >= 1"

  out = []
  w = out.append
  w(f"// Stress test: array size {n}, {n} array_get ops with constant indices.")
  w("")

  out_ports = ", ".join(f"out z{i} : i1" for i in range(n))
  w(f"hw.module @ConcatCanonicalizationScaling{n}(in %in: !hw.array<{n}xi1>, {out_ports}) {{"
   )
  index_bits = (n - 1).bit_length()
  for i in range(n):
    w(f"  %c{i} = hw.constant {i} : i{index_bits}")
    w(f"  %v{i} = hw.array_get %in[%c{i}] : !hw.array<{n}xi1>, i{index_bits}")

  out_args = ", ".join(f"%v{i}" for i in range(n))
  out_types = ", ".join(["i1"] * n)
  w(f"  hw.output {out_args} : {out_types}")
  w("}")
  return "\n".join(out) + "\n"


if __name__ == "__main__":
  if len(sys.argv) != 2:
    sys.stderr.write(f"usage: {sys.argv[0]} N\n")
    sys.exit(2)
  sys.stdout.write(generate(int(sys.argv[1])))
