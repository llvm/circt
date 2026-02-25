// RUN: circt-bmc %s --module=ltl_ok -b 2 --emit-mlir | FileCheck %s
// CHECK: llvm.func @printf
// CHECK-NOT: ltl.

hw.module @ltl_ok(in %clk: !seq.clock, in %req: i1, in %ack: i1) attributes {num_regs = 0 : i32, initial_values = []} {
  %seq = ltl.delay %ack, 2, 0 : i1
  %imp = ltl.implication %req, %seq : i1, !ltl.sequence
  %clk_i1 = seq.from_clock %clk
  %prop = ltl.clock %imp, posedge %clk_i1 : !ltl.property
  verif.assert %prop : !ltl.property
  hw.output
}
