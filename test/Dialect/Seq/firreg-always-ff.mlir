// Opt-in `always_ff` lowering of seq.firreg (default emission is plain `always`;
// see firreg-clock-reset-attrs.mlir). The `emit-firreg-always-ff` option routes
// register logic through `sv.alwaysff`, which expresses the clock edge, reset
// kind, and reset edge directly. Active-low async resets render as
// `always_ff @(... or negedge rst) if (!rst) ...` structurally, with no inverted
// temporary.
// RUN: circt-opt %s --pass-pipeline="builtin.module(lower-seq-to-sv{disable-reg-randomization emit-firreg-always-ff})" | FileCheck %s --check-prefix=MLIR
// RUN: circt-opt %s --lower-seq-to-sv="disable-reg-randomization emit-firreg-always-ff" --export-verilog | FileCheck %s --check-prefix=SV

// Row 1: posedge clock, sync active-high reset.
// MLIR-LABEL: hw.module @PosSyncHigh
// MLIR: sv.alwaysff(posedge %clk) {
// MLIR: }(syncreset : posedge %rst) {
// SV-LABEL: module PosSyncHigh
// SV: always_ff @(posedge clk) begin
// SV-NEXT: if (rst)
// SV-NEXT: r <= 32'h0;
// SV: else
// SV-NEXT: r <= d;
hw.module @PosSyncHigh(in %clk : !seq.clock, in %rst : i1, in %d : i32) {
  %c0 = hw.constant 0 : i32
  %r = seq.firreg %d clock %clk reset sync %rst, %c0 {clockEdge = 0 : i32, resetPolarity = 0 : i32} : i32
}

// Row 2: posedge clock, sync active-low reset -> `if (!rst)` on the original
// signal, no inverted temporary (the `sv.alwaysff` emitter inverts the condition
// for an active-low/`AtNegEdge` reset of any reset style).
// SV-LABEL: module PosSyncLow
// SV-NOT: = !rst
// SV: always_ff @(posedge clk) begin
// SV-NEXT: if (!rst)
// SV-NEXT: r <= 32'h0;
hw.module @PosSyncLow(in %clk : !seq.clock, in %rst : i1, in %d : i32) {
  %c0 = hw.constant 0 : i32
  %r = seq.firreg %d clock %clk reset sync %rst, %c0 {clockEdge = 0 : i32, resetPolarity = 1 : i32} : i32
}

// Row 3: posedge clock, async active-high reset.
// SV-LABEL: module PosAsyncHigh
// SV: always_ff @(posedge clk or posedge rst) begin
// SV-NEXT: if (rst)
// SV-NEXT: r <= 32'h0;
hw.module @PosAsyncHigh(in %clk : !seq.clock, in %rst : i1, in %d : i32) {
  %c0 = hw.constant 0 : i32
  %r = seq.firreg %d clock %clk reset async %rst, %c0 {clockEdge = 0 : i32, resetPolarity = 0 : i32} : i32
}

// Row 4: posedge clock, async active-low reset -- the behavior-changing case.
// Triggers on `negedge rst` (original signal) and tests `if (!rst)` structurally.
// MLIR-LABEL: hw.module @PosAsyncLow
// MLIR: sv.alwaysff(posedge %clk) {
// MLIR: }(asyncreset : negedge %rst) {
// SV-LABEL: module PosAsyncLow
// SV: always_ff @(posedge clk or negedge rst) begin
// SV-NEXT: if (!rst)
// SV-NEXT: r <= 32'h0;
hw.module @PosAsyncLow(in %clk : !seq.clock, in %rst : i1, in %d : i32) {
  %c0 = hw.constant 0 : i32
  %r = seq.firreg %d clock %clk reset async %rst, %c0 {clockEdge = 0 : i32, resetPolarity = 1 : i32} : i32
}

// Row 5: negedge clock, sync active-high reset.
// SV-LABEL: module NegSyncHigh
// SV: always_ff @(negedge clk) begin
// SV-NEXT: if (rst)
hw.module @NegSyncHigh(in %clk : !seq.clock, in %rst : i1, in %d : i32) {
  %c0 = hw.constant 0 : i32
  %r = seq.firreg %d clock %clk reset sync %rst, %c0 {clockEdge = 1 : i32, resetPolarity = 0 : i32} : i32
}

// Row 6: negedge clock, sync active-low reset -> `if (!rst)`, no inverted temp.
// SV-LABEL: module NegSyncLow
// SV-NOT: = !rst
// SV: always_ff @(negedge clk) begin
// SV-NEXT: if (!rst)
hw.module @NegSyncLow(in %clk : !seq.clock, in %rst : i1, in %d : i32) {
  %c0 = hw.constant 0 : i32
  %r = seq.firreg %d clock %clk reset sync %rst, %c0 {clockEdge = 1 : i32, resetPolarity = 1 : i32} : i32
}

// Row 7: negedge clock, async active-high reset.
// SV-LABEL: module NegAsyncHigh
// SV: always_ff @(negedge clk or posedge rst) begin
// SV-NEXT: if (rst)
hw.module @NegAsyncHigh(in %clk : !seq.clock, in %rst : i1, in %d : i32) {
  %c0 = hw.constant 0 : i32
  %r = seq.firreg %d clock %clk reset async %rst, %c0 {clockEdge = 1 : i32, resetPolarity = 0 : i32} : i32
}

// Row 8: negedge clock, async active-low reset.
// SV-LABEL: module NegAsyncLow
// SV: always_ff @(negedge clk or negedge rst) begin
// SV-NEXT: if (!rst)
hw.module @NegAsyncLow(in %clk : !seq.clock, in %rst : i1, in %d : i32) {
  %c0 = hw.constant 0 : i32
  %r = seq.firreg %d clock %clk reset async %rst, %c0 {clockEdge = 1 : i32, resetPolarity = 1 : i32} : i32
}

// Row 9: posedge clock, no reset.
// MLIR-LABEL: hw.module @PosNoReset
// MLIR: sv.alwaysff(posedge %clk) {
// SV-LABEL: module PosNoReset
// SV: always_ff @(posedge clk)
// SV-NEXT: r <= d;
hw.module @PosNoReset(in %clk : !seq.clock, in %d : i32) {
  %r = seq.firreg %d clock %clk {clockEdge = 0 : i32} : i32
}

// Row 10: negedge clock, no reset.
// SV-LABEL: module NegNoReset
// SV: always_ff @(negedge clk)
// SV-NEXT: r <= d;
hw.module @NegNoReset(in %clk : !seq.clock, in %d : i32) {
  %r = seq.firreg %d clock %clk {clockEdge = 1 : i32} : i32
}

// Compound (non-leaf) active-low reset expression: the inverted `always_ff`
// condition must be `!(a & b)`, not `!a & b` (== `(!a) & b`). ExportVerilog
// spills the compound reset to a wire, so the block tests `if (!_GEN)` on that
// leaf wire, and the emitter additionally prints the operand at unary precedence
// so any non-spilled compound operand is parenthesized.
// SV-LABEL: module CompoundActiveLowReset
// SV: wire _GEN = a & b;
// SV: always_ff @(posedge clk or negedge _GEN) begin
// SV-NEXT: if (!_GEN)
hw.module @CompoundActiveLowReset(in %clk : !seq.clock, in %d : i32, in %a : i1, in %b : i1) {
  %rst = comb.and %a, %b : i1
  %c0 = hw.constant 0 : i32
  %r = seq.firreg %d clock %clk reset async %rst, %c0 {clockEdge = 0 : i32, resetPolarity = 1 : i32} : i32
}
