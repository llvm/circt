# RUN: %PYTHON% %s %t | FileCheck %s

from pycde import (Clock, Output, Input, Module, generator, System)
from pycde.dialects import comb
from pycde.testing import unittestmodule
from pycde.types import Bits

import sys

# Test that debug mode inserts hw.wire with symbols that survive optimization
# passes and appear as named wires in the output SystemVerilog.
#
# The generator below uses `or x, 0` and `and x, 0xFF` which are trivially
# canonicalized to identity in normal mode.  In debug mode the auto-named
# hw.wire ops carry inner symbols that prevent removal, so every intermediate
# Python variable name appears as a wire in the final Verilog.


class DebugWireTest(Module):
  clk = Clock()
  a = Input(Bits(8))
  b = Input(Bits(8))
  out = Output(Bits(8))

  @generator
  def build(ports):
    # Intermediate value — should appear as a named wire in debug Verilog.
    sum_ab = comb.AddOp(ports.a, ports.b)

    # `or x, 0` is identity and normally canonicalized away.
    zero = Bits(8)(0)
    kept = comb.OrOp(sum_ab, zero)

    # `and x, 0xFF` is identity and normally canonicalized away.
    mask = Bits(8)(0xFF)
    also_kept = comb.AndOp(kept, mask)

    ports.out = also_kept


s = System([DebugWireTest], output_directory=sys.argv[1], debug=True)
s.generate()

# --- Check MLIR before passes: hw.wire ops present with syms ---
# CHECK-LABEL: hw.module @DebugWireTest
# CHECK:         [[ADD:%.+]] = comb.add bin %a, %b : i8
# CHECK:         %sum_ab = hw.wire [[ADD]] sym @sum_ab
# CHECK:         [[OR:%.+]] = comb.or bin %sum_ab,
# CHECK:         %kept = hw.wire [[OR]] sym @kept
# CHECK:         [[AND:%.+]] = comb.and bin %kept,
# CHECK:         %also_kept = hw.wire [[AND]] sym @also_kept
# CHECK:         hw.output %also_kept
s.print()

s.run_passes()

# --- Check MLIR after passes: hw.wire ops survived canonicalization ---
# CHECK-LABEL: hw.module @DebugWireTest
# CHECK:         %sum_ab = hw.wire {{%.+}} sym @sum_ab
# CHECK:         %kept = hw.wire %sum_ab sym @kept
# CHECK:         %also_kept = hw.wire %kept sym @also_kept
# CHECK:         hw.output %also_kept
s.print()

s.emit_outputs()

# --- Check SystemVerilog output contains the named wires ---
import os

sv_path = os.path.join(sys.argv[1], "hw", "DebugWireTest.sv")
with open(sv_path) as f:
  sv = f.read()
  print(sv)
# CHECK-LABEL: module DebugWireTest
# CHECK:         wire [7:0] sum_ab = a + b;
# CHECK:         wire [7:0] kept = sum_ab;
# CHECK:         wire [7:0] also_kept = kept;
# CHECK:         assign out = also_kept;
