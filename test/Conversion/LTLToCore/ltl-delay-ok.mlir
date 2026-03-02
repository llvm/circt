// RUN: circt-opt %s --lower-ltl-to-core | FileCheck %s

// CHECK-LABEL: hw.module @ltl_ok
// CHECK-SAME: in %clk : !seq.clock, in %req : i1, in %ack : i1
hw.module @ltl_ok(in %clk: !seq.clock, in %req: i1, in %ack: i1) attributes {num_regs = 0 : i32, initial_values = []} {
  %seq = ltl.delay %ack, 2, 0 : i1
  %imp = ltl.implication %req, %seq : i1, !ltl.sequence
  %clk_i1 = seq.from_clock %clk
  %prop = ltl.clock %imp, posedge %clk_i1 : !ltl.property

  // CHECK: %ltl_delay_0 = seq.compreg sym @ltl_delay_0 %req, %clk
  // CHECK: %ltl_delay_1 = seq.compreg sym @ltl_delay_1 %ltl_delay_0, %clk
  // CHECK: [[TRUE:%.+]] = hw.constant true
  // CHECK: [[NOT_DELAYED:%.+]] = comb.xor %ltl_delay_1, [[TRUE]] : i1
  // CHECK: [[OR:%.+]] = comb.or [[NOT_DELAYED]], %ack : i1
  // CHECK: verif.assert [[OR]] : i1
  verif.assert %prop : !ltl.property
  hw.output
}
