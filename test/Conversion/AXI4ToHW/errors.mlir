// RUN: circt-opt %s --lower-axi4-to-hw --verify-diagnostics

// The pass is not implemented yet, so it rejects any input
// expected-error @below {{lowering AXI4 to HW is not yet implemented}}
module {
  hw.module @NoAXI4(in %clk : !seq.clock, in %rst_ni : i1, in %a : i8, out b : i8) {
    hw.output %a : i8
  }
}
