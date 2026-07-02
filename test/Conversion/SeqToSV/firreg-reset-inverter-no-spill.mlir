// The reset-polarity inverter (`sv.resetInverter`) is never spilled to a wire,
// even under an aggressive `maximumNumberOfTermsPerExpression` that would
// otherwise force every sub-term out-of-line. This makes the active-low async
// reset race-free as a STRUCTURAL guarantee (not a default-config coincidence):
// the sensitivity-list `negedge _GEN` and the clocked `if (!_GEN)` stay anchored
// to the same single reset net `_GEN`, with no second inverted wire that would
// lag the sensitivity edge by a delta cycle.
//
// RUN: circt-opt %s --lower-seq-to-sv="disable-reg-randomization" --cse --test-apply-lowering-options='options=maximumNumberOfTermsPerExpression=0' --export-verilog | FileCheck %s

// CHECK-LABEL: module NonLeafAsyncLow
// CHECK: wire _GEN = a & b;
// CHECK-NOT: = !_GEN
// CHECK-NOT: = ~_GEN
// CHECK: always @(posedge clk or negedge _GEN) begin
// CHECK-NEXT: if (!_GEN)
// CHECK-NEXT: r <= 32'h0;
hw.module @NonLeafAsyncLow(in %clk: !seq.clock, in %d: i32, in %a: i1, in %b: i1, out q: i32) {
  %rst = comb.and %a, %b : i1
  %c0 = hw.constant 0 : i32
  %r = seq.firreg %d clock %clk reset async %rst, %c0 {clockEdge = 0 : i32, resetPolarity = 1 : i32} : i32
  hw.output %r : i32
}
