// RUN: not circt-bmc %s --module=ltl_bad -b 2 --emit-mlir 2>&1 | FileCheck %s
// CHECK: error: unsupported LTL pattern for circt-bmc lowering

hw.module @ltl_bad(in %clk: !seq.clock, in %req: i1, in %ack: i1) attributes {num_regs = 0 : i32, initial_values = []} {
  // Unbounded delay is not supported by lower-ltl-to-bmc.
  %seq = ltl.delay %ack, 2 : i1
  %imp = ltl.implication %req, %seq : i1, !ltl.sequence
  %clk_i1 = seq.from_clock %clk
  %prop = ltl.clock %imp, posedge %clk_i1 : !ltl.property
  verif.assert %prop : !ltl.property
  hw.output
}
