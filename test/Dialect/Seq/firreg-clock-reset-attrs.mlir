// Lowering of the seq.firreg clock-edge and reset-polarity attributes to SV.
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

// An active-low (NegReset) synchronous reset inverts the reset condition and
// keeps a posedge clock: it resets when the original reset is low.
// CHECK-LABEL: hw.module @SyncActiveLow
// CHECK: %[[INV:.+]] = comb.xor %rst, %true{{.*}}sv.resetInverter
// CHECK: sv.always posedge %clk {
// CHECK-NEXT: sv.if %[[INV]] {
// ROUNDTRIP-LABEL: hw.module @SyncActiveLow
// ROUNDTRIP: seq.firreg %in clock %clk reset sync %rst, %c0_i32 {clockEdge = 0 : i32, resetPolarity = 1 : i32}
hw.module @SyncActiveLow(in %clk : !seq.clock, in %rst : i1, in %in : i32) {
  %c0 = hw.constant 0 : i32
  %r = seq.firreg %in clock %clk reset sync %rst, %c0 {clockEdge = 0 : i32, resetPolarity = 1 : i32} : i32
}

// An active-low (NegReset) asynchronous reset triggers on the configured edge of
// the ORIGINAL reset signal -- `negedge %rst` -- and tests the inline inverted
// condition `!rst`, i.e. `always @(posedge clk or negedge rst) if (!rst) <reset>
// else <next>`, with no manufactured inverted temporary in the sensitivity list.
// ExportVerilog duplicates the 1-bit `!rst` inline at each use (the NOT-of-leaf
// case of isDuplicatableExpression), so a post-lowering CSE cannot spill it to a
// wire that would race against the raw-`rst` sensitivity edge. `if (!rst)` is
// X-correct (an X reset takes the non-reset branch) and IfOp::canonicalize never
// flips a populated reset-if.
// CHECK-LABEL: hw.module @AsyncActiveLow
// CHECK: %[[INV:.+]] = comb.xor %rst, %true{{.*}}sv.resetInverter
// CHECK: sv.always posedge %clk, negedge %rst {
// CHECK-NEXT: sv.if %[[INV]] {
// CHECK-NEXT: sv.passign %{{.+}}, %c0_i32
// CHECK: } else {
// CHECK-NEXT: sv.passign %{{.+}}, %in
// ROUNDTRIP-LABEL: hw.module @AsyncActiveLow
// ROUNDTRIP: seq.firreg %in clock %clk reset async %rst, %c0_i32 {clockEdge = 0 : i32, resetPolarity = 1 : i32}
hw.module @AsyncActiveLow(in %clk : !seq.clock, in %rst : i1, in %in : i32) {
  %c0 = hw.constant 0 : i32
  %r = seq.firreg %in clock %clk reset async %rst, %c0 {clockEdge = 0 : i32, resetPolarity = 1 : i32} : i32
}

// An active-high async reset keeps the posedge sensitivity and an un-inverted
// condition; clock edge and reset kind round-trip through the enum reprs.
// CHECK-LABEL: hw.module @AsyncActiveHigh
// CHECK: sv.always posedge %clk, posedge %rst {
// CHECK-NEXT: sv.if %rst {
// ROUNDTRIP-LABEL: hw.module @AsyncActiveHigh
// ROUNDTRIP: seq.firreg %in clock %clk reset async %rst, %c0_i32 {clockEdge = 0 : i32, resetPolarity = 0 : i32} : i32
hw.module @AsyncActiveHigh(in %clk : !seq.clock, in %rst : i1, in %in : i32) {
  %c0 = hw.constant 0 : i32
  %r = seq.firreg %in clock %clk reset async %rst, %c0 {clockEdge = 0 : i32, resetPolarity = 0 : i32} : i32
}
