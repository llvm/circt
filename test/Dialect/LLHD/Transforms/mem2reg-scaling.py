#!/usr/bin/env python3
# Generate an LLHD mem2reg scaling stress test.
#
# The produced MLIR module defines N signals and a process that probes all of
# them, then drives each signal inside a chain of N/2 if/else diamonds whose
# joins become 2-predecessor merge points. Then-branches use blocking drives,
# else-branches use delta drives, so the mem2reg lattice has to reconcile both
# flavors at every join. The control flow loops back to the probes, creating
# back-edges that exercise the iterative data-flow solver.

import sys


def generate(n: int) -> str:
  assert n >= 2 and n % 2 == 0, "N must be an even integer >= 2"
  num_diamonds = n // 2

  out = []
  w = out.append
  w(f"// Stress test with {n} signals and {num_diamonds} if/else diamonds.")
  w("")
  w(f"hw.module @Scaling{n}(in %u: i42, in %cond: i1) {{")
  w("  %td = llhd.constant_time <0ns, 1d, 0e>")
  w("  %tb = llhd.constant_time <0ns, 0d, 1e>")
  for i in range(n):
    w(f"  %a{i} = llhd.sig %u : i42")
  w("  llhd.process {")
  w("    cf.br ^body")
  w("  ^body:")
  for i in range(n):
    w(f"    %p{i} = llhd.prb %a{i} : i42")
  w("    cf.cond_br %cond, ^t0, ^e0")
  for d in range(num_diamonds):
    w(f"  ^t{d}:")
    w(f"    llhd.drv %a{2 * d}, %u after %tb : i42")
    w(f"    cf.br ^j{d}")
    w(f"  ^e{d}:")
    w(f"    llhd.drv %a{2 * d + 1}, %u after %td : i42")
    w(f"    cf.br ^j{d}")
    w(f"  ^j{d}:")
    if d + 1 < num_diamonds:
      w(f"    cf.cond_br %cond, ^t{d + 1}, ^e{d + 1}")
    else:
      w("    cf.cond_br %cond, ^body, ^exit")
  w("  ^exit:")
  w("    llhd.halt")
  w("  }")
  w("}")
  return "\n".join(out) + "\n"


if __name__ == "__main__":
  if len(sys.argv) != 2:
    sys.stderr.write(f"usage: {sys.argv[0]} N\n")
    sys.exit(2)
  sys.stdout.write(generate(int(sys.argv[1])))
