// Lowering of the seq.firreg clock-edge attribute to SV.
// RUN: circt-opt %s --pass-pipeline="builtin.module(lower-seq-to-sv{disable-reg-randomization})" | FileCheck %s
// RUN: circt-opt %s | circt-opt | FileCheck %s --check-prefix=ROUNDTRIP

// A non-default (negedge) clock edge lowers to a negedge-triggered always block.
// CHECK-LABEL: hw.module @NegEdgeClock
// CHECK: sv.always negedge %clk {
// ROUNDTRIP-LABEL: hw.module @NegEdgeClock
// ROUNDTRIP: seq.firreg %in clock %clk {clockEdge = 1 : i32}
hw.module @NegEdgeClock(in %clk : !seq.clock, in %in : i32) {
  %r = seq.firreg %in clock %clk {clockEdge = 1 : i32} : i32
}
