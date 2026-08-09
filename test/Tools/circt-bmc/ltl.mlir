// RUN: circt-bmc %s -b 6 --module LtlProbe --emit-mlir -o - | FileCheck %s --implicit-check-not=ltl.

// CHECK-LABEL: func.func @bmc_circuit(
// CHECK-SAME:      %[[CLOCK:[^:]+]]: !smt.bv<1>,
// CHECK-SAME:      %[[A:[^:]+]]: !smt.bv<1>,
// CHECK-SAME:      %[[B:[^:]+]]: !smt.bv<1>,
// CHECK-SAME:      %[[SAMPLED_A:[^:]+]]: !smt.bv<1>,
// CHECK-SAME:      %[[SAMPLED_B:[^:]+]]: !smt.bv<1>,
// CHECK-SAME:      %[[PAST_A_1:[^:]+]]: !smt.bv<1>,
// CHECK-SAME:      %[[PAST_A_2:[^:]+]]: !smt.bv<1>,
// CHECK-SAME:      %[[VALID_0:[^:]+]]: !smt.bv<1>,
// CHECK-SAME:      %[[VALID_1:[^:]+]]: !smt.bv<1>,
// CHECK-SAME:      %[[VALID_2:[^:]+]]: !smt.bv<1>)
// CHECK:         %[[TRUE:.+]] = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK:         %[[NOT_A:.+]] = smt.bv.xor %[[PAST_A_2]], %[[TRUE]] : !smt.bv<1>
// CHECK:         %[[IMPLICATION:.+]] = smt.bv.or %[[NOT_A]], %[[SAMPLED_B]] : !smt.bv<1>
// CHECK:         %[[NOT_VALID:.+]] = smt.bv.xor %[[VALID_2]], %[[TRUE]] : !smt.bv<1>
// CHECK:         %[[GUARDED:.+]] = smt.bv.or %[[NOT_VALID]], %[[IMPLICATION]] : !smt.bv<1>
// CHECK:         %[[HOLDS:.+]] = smt.eq %[[GUARDED]], %[[TRUE]] : !smt.bv<1>
// CHECK:         %[[VIOLATED:.+]] = smt.not %[[HOLDS]]
// CHECK:         smt.assert %[[VIOLATED]]
// CHECK:         return %[[A]], %[[B]], %[[SAMPLED_A]], %[[PAST_A_1]], %[[TRUE]], %[[VALID_0]], %[[VALID_1]]

hw.module @LtlProbe(in %clock : !seq.clock, in %a : i1, in %b : i1) {
  %0 = seq.from_clock %clock
  %1 = ltl.clocked_atom %a, posedge %0 : i1
  %2 = ltl.clocked_atom %b, posedge %0 : i1
  %3 = ltl.clocked_delay %2, posedge %0, 2, 0 : !ltl.sequence
  %4 = ltl.implication %1, %3 : !ltl.sequence, !ltl.sequence
  verif.assert %4 label "ltl_probe" : !ltl.property
  hw.output
}
