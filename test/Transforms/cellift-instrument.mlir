// RUN: circt-opt --cellift-instrument --split-input-file %s | FileCheck %s

// -----
// Test: Constants get zero taint by default.
// CHECK-LABEL: hw.module @test_constant
// CHECK-SAME: (out y : i8, out y_t : i8)
hw.module @test_constant(out y : i8) {
  // CHECK: %c42_i8 = hw.constant 42 : i8
  // CHECK: %c0_i8 = hw.constant 0 : i8
  // CHECK: hw.output %c42_i8, %c0_i8 : i8, i8
  %c42 = hw.constant 42 : i8
  hw.output %c42 : i8
}

// -----
// Test: Module input ports get taint ports added.
// CHECK-LABEL: hw.module @test_passthrough
// CHECK-SAME: (in %a : i8, in %a_t : i8, out y : i8, out y_t : i8)
hw.module @test_passthrough(in %a : i8, out y : i8) {
  // CHECK: hw.output %a, %a_t : i8, i8
  hw.output %a : i8
}

// -----
// Test: comb.and taint rule: y_t = (a & b_t) | (b & a_t) | (a_t & b_t)
// CHECK-LABEL: hw.module @test_and
// CHECK-SAME: (in %a : i8, in %a_t : i8, in %b : i8, in %b_t : i8, out y : i8, out y_t : i8)
hw.module @test_and(in %a : i8, in %b : i8, out y : i8) {
  // CHECK: %0 = comb.and %a, %b : i8
  // CHECK: [[T1:%.+]] = comb.and %a, %b_t : i8
  // CHECK: [[T2:%.+]] = comb.and %b, %a_t : i8
  // CHECK: [[T3:%.+]] = comb.and %a_t, %b_t : i8
  // CHECK: [[Y_T:%.+]] = comb.or [[T1]], [[T2]], [[T3]] : i8
  %0 = comb.and %a, %b : i8
  hw.output %0 : i8
}

// -----
// Test: comb.or taint rule: y_t = (~a & b_t) | (~b & a_t) | (a_t & b_t)
// CHECK-LABEL: hw.module @test_or
// CHECK-SAME: (in %a : i8, in %a_t : i8, in %b : i8, in %b_t : i8, out y : i8, out y_t : i8)
hw.module @test_or(in %a : i8, in %b : i8, out y : i8) {
  // CHECK: %0 = comb.or %a, %b : i8
  // CHECK: %c-1_i8 = hw.constant -1 : i8
  // CHECK: [[NOT_A:%.+]] = comb.xor %a, %c-1_i8 : i8
  // CHECK: [[NOT_B:%.+]] = comb.xor %b, %c-1_i8 : i8
  // CHECK: [[T1:%.+]] = comb.and [[NOT_A]], %b_t : i8
  // CHECK: [[T2:%.+]] = comb.and [[NOT_B]], %a_t : i8
  // CHECK: [[T3:%.+]] = comb.and %a_t, %b_t : i8
  // CHECK: [[Y_T:%.+]] = comb.or [[T1]], [[T2]], [[T3]] : i8
  %0 = comb.or %a, %b : i8
  hw.output %0 : i8
}

// -----
// Test: comb.xor taint rule: y_t = a_t | b_t
// CHECK-LABEL: hw.module @test_xor
// CHECK-SAME: (in %a : i8, in %a_t : i8, in %b : i8, in %b_t : i8, out y : i8, out y_t : i8)
hw.module @test_xor(in %a : i8, in %b : i8, out y : i8) {
  // CHECK: %0 = comb.xor %a, %b : i8
  // CHECK: [[Y_T:%.+]] = comb.or %a_t, %b_t : i8
  %0 = comb.xor %a, %b : i8
  hw.output %0 : i8
}

// -----
// Test: comb.add taint rule (precise): y_t = ((a&~a_t)+(b&~b_t)) XOR ((a|a_t)+(b|b_t)) | a_t | b_t
// CHECK-LABEL: hw.module @test_add
// CHECK-SAME: (in %a : i8, in %a_t : i8, in %b : i8, in %b_t : i8, out y : i8, out y_t : i8)
hw.module @test_add(in %a : i8, in %b : i8, out y : i8) {
  // CHECK: %0 = comb.add %a, %b : i8
  // CHECK: %c-1_i8 = hw.constant -1 : i8
  // CHECK: [[NOT_AT:%.+]] = comb.xor %a_t, %c-1_i8 : i8
  // CHECK: [[NOT_BT:%.+]] = comb.xor %b_t, %c-1_i8 : i8
  // CHECK: [[A_ZERO:%.+]] = comb.and %a, [[NOT_AT]] : i8
  // CHECK: [[B_ZERO:%.+]] = comb.and %b, [[NOT_BT]] : i8
  // CHECK: [[A_ONE:%.+]] = comb.or %a, %a_t : i8
  // CHECK: [[B_ONE:%.+]] = comb.or %b, %b_t : i8
  // CHECK: [[SUM_MIN:%.+]] = comb.add [[A_ZERO]], [[B_ZERO]] : i8
  // CHECK: [[SUM_MAX:%.+]] = comb.add [[A_ONE]], [[B_ONE]] : i8
  // CHECK: [[XOR:%.+]] = comb.xor [[SUM_MIN]], [[SUM_MAX]] : i8
  // CHECK: [[Y_T:%.+]] = comb.or [[XOR]], %a_t, %b_t : i8
  %0 = comb.add %a, %b : i8
  hw.output %0 : i8
}

// -----
// Test: comb.sub taint rule (precise): y_t = ((a|a_t)-(b&~b_t)) XOR ((a&~a_t)-(b|b_t)) | a_t | b_t
// CHECK-LABEL: hw.module @test_sub
// CHECK-SAME: (in %a : i8, in %a_t : i8, in %b : i8, in %b_t : i8, out y : i8, out y_t : i8)
hw.module @test_sub(in %a : i8, in %b : i8, out y : i8) {
  // CHECK: comb.sub %a, %b : i8
  // CHECK: comb.xor %a_t, %c-1_i8 : i8
  // CHECK: comb.xor %b_t, %c-1_i8 : i8
  // CHECK: comb.or %a, %a_t : i8
  // CHECK: comb.and %b, {{%.+}} : i8
  // CHECK: comb.and %a, {{%.+}} : i8
  // CHECK: comb.or %b, %b_t : i8
  // CHECK: [[S1:%.+]] = comb.sub {{%.+}}, {{%.+}} : i8
  // CHECK: [[S2:%.+]] = comb.sub {{%.+}}, {{%.+}} : i8
  // CHECK: [[XOR:%.+]] = comb.xor [[S1]], [[S2]] : i8
  // CHECK: [[Y_T:%.+]] = comb.or [[XOR]], %a_t, %b_t : i8
  %0 = comb.sub %a, %b : i8
  hw.output %0 : i8
}

// -----
// Test: comb.mul taint rule (conservative).
// CHECK-LABEL: hw.module @test_mul
// CHECK-SAME: (in %a : i8, in %a_t : i8, in %b : i8, in %b_t : i8, out y : i8, out y_t : i8)
hw.module @test_mul(in %a : i8, in %b : i8, out y : i8) {
  // CHECK: comb.mul %a, %b : i8
  // CHECK: comb.icmp ne %a_t
  // CHECK: comb.icmp ne %b_t
  // CHECK: comb.or
  // CHECK: comb.replicate
  %0 = comb.mul %a, %b : i8
  hw.output %0 : i8
}

// -----
// Test: comb.mux taint rule.
// y_t = mux(sel, t_t, f_t) | replicate(sel_t) & (t ^ f | t_t | f_t)
// CHECK-LABEL: hw.module @test_mux
// CHECK-SAME: (in %sel : i1, in %sel_t : i1, in %a : i8, in %a_t : i8, in %b : i8, in %b_t : i8, out y : i8, out y_t : i8)
hw.module @test_mux(in %sel : i1, in %a : i8, in %b : i8, out y : i8) {
  // CHECK: %0 = comb.mux %sel, %a, %b : i8
  // CHECK: [[DATA_T:%.+]] = comb.mux %sel, %a_t, %b_t : i8
  // CHECK: [[SEL_B:%.+]] = comb.replicate %sel_t : (i1) -> i8
  // CHECK: [[DIFF:%.+]] = comb.xor %a, %b : i8
  // CHECK: [[INNER:%.+]] = comb.or [[DIFF]], %a_t, %b_t : i8
  // CHECK: [[CTRL:%.+]] = comb.and [[SEL_B]], [[INNER]] : i8
  // CHECK: [[Y_T:%.+]] = comb.or [[DATA_T]], [[CTRL]] : i8
  %0 = comb.mux %sel, %a, %b : i8
  hw.output %0 : i8
}

// -----
// Test: comb.concat taint rule.
// CHECK-LABEL: hw.module @test_concat
// CHECK-SAME: (in %a : i4, in %a_t : i4, in %b : i4, in %b_t : i4, out y : i8, out y_t : i8)
hw.module @test_concat(in %a : i4, in %b : i4, out y : i8) {
  // CHECK: %0 = comb.concat %a, %b : i4, i4
  // CHECK: [[Y_T:%.+]] = comb.concat %a_t, %b_t : i4, i4
  %0 = comb.concat %a, %b : i4, i4
  hw.output %0 : i8
}

// -----
// Test: comb.extract taint rule.
// CHECK-LABEL: hw.module @test_extract
// CHECK-SAME: (in %a : i8, in %a_t : i8, out y : i4, out y_t : i4)
hw.module @test_extract(in %a : i8, out y : i4) {
  // CHECK: %0 = comb.extract %a from 2 : (i8) -> i4
  // CHECK: [[Y_T:%.+]] = comb.extract %a_t from 2 : (i8) -> i4
  %0 = comb.extract %a from 2 : (i8) -> i4
  hw.output %0 : i4
}

// -----
// Test: comb.replicate taint rule.
// CHECK-LABEL: hw.module @test_replicate
// CHECK-SAME: (in %a : i4, in %a_t : i4, out y : i8, out y_t : i8)
hw.module @test_replicate(in %a : i4, out y : i8) {
  // CHECK: %0 = comb.replicate %a : (i4) -> i8
  // CHECK: [[Y_T:%.+]] = comb.replicate %a_t : (i4) -> i8
  %0 = comb.replicate %a : (i4) -> i8
  hw.output %0 : i8
}

// -----
// Test: comb.icmp eq precise taint rule:
// y_t = has_taint & (masked_a == masked_b) where mask = ~(a_t | b_t)
// CHECK-LABEL: hw.module @test_icmp
// CHECK-SAME: (in %a : i8, in %a_t : i8, in %b : i8, in %b_t : i8, out y : i1, out y_t : i1)
hw.module @test_icmp(in %a : i8, in %b : i8, out y : i1) {
  // CHECK: %0 = comb.icmp eq %a, %b : i8
  // CHECK: [[COMBINED:%.+]] = comb.or %a_t, %b_t : i8
  // CHECK: [[HAS_T:%.+]] = comb.icmp ne [[COMBINED]]
  // CHECK: [[MASK:%.+]] = comb.xor [[COMBINED]], %c-1_i8 : i8
  // CHECK: [[MA:%.+]] = comb.and %a, [[MASK]] : i8
  // CHECK: [[MB:%.+]] = comb.and %b, [[MASK]] : i8
  // CHECK: [[EQ_UN:%.+]] = comb.icmp eq [[MA]], [[MB]] : i8
  // CHECK: [[Y_T:%.+]] = comb.and [[HAS_T]], [[EQ_UN]] : i1
  %0 = comb.icmp eq %a, %b : i8
  hw.output %0 : i1
}

// -----
// Test: comb.shl taint rule (current conservative implementation).
// CHECK-LABEL: hw.module @test_shl
// CHECK-SAME: (in %a : i8, in %a_t : i8, in %b : i8, in %b_t : i8, out y : i8, out y_t : i8)
hw.module @test_shl(in %a : i8, in %b : i8, out y : i8) {
  // CHECK: comb.shl %a, %b : i8
  // CHECK: [[HAS_BT:%.+]] = comb.icmp ne %b_t, %c0_i8 : i8
  // CHECK: [[AMT_T:%.+]] = comb.replicate [[HAS_BT]] : (i1) -> i8
  // CHECK: [[SHIFTED_T:%.+]] = comb.shl %a_t, %b : i8
  // CHECK: [[Y_T:%.+]] = comb.or [[AMT_T]], [[SHIFTED_T]] : i8
  %0 = comb.shl %a, %b : i8
  hw.output %0 : i8
}

// -----
// Test: comb.shru taint rule (current conservative shift fallback).
// CHECK-LABEL: hw.module @test_shru
// CHECK-SAME: (in %a : i8, in %a_t : i8, in %b : i8, in %b_t : i8, out y : i8, out y_t : i8)
hw.module @test_shru(in %a : i8, in %b : i8, out y : i8) {
  // CHECK: comb.shru %a, %b : i8
  // CHECK: [[HAS_BT:%.+]] = comb.icmp ne %b_t, %c0_i8 : i8
  // CHECK: [[AMT_T:%.+]] = comb.replicate [[HAS_BT]] : (i1) -> i8
  // CHECK: [[SHIFTED_T:%.+]] = comb.shru %a_t, %b : i8
  // CHECK: [[Y_T:%.+]] = comb.or [[AMT_T]], [[SHIFTED_T]] : i8
  %0 = comb.shru %a, %b : i8
  hw.output %0 : i8
}

// -----
// Test: comb.shrs taint rule (current conservative shift fallback).
// CHECK-LABEL: hw.module @test_shrs
// CHECK-SAME: (in %a : i8, in %a_t : i8, in %b : i8, in %b_t : i8, out y : i8, out y_t : i8)
hw.module @test_shrs(in %a : i8, in %b : i8, out y : i8) {
  // CHECK: comb.shrs %a, %b : i8
  // CHECK: [[HAS_BT:%.+]] = comb.icmp ne %b_t, %c0_i8 : i8
  // CHECK: [[AMT_T:%.+]] = comb.replicate [[HAS_BT]] : (i1) -> i8
  // CHECK: [[SHIFTED_T:%.+]] = comb.shrs %a_t, %b : i8
  // CHECK: [[Y_T:%.+]] = comb.or [[AMT_T]], [[SHIFTED_T]] : i8
  %0 = comb.shrs %a, %b : i8
  hw.output %0 : i8
}

// -----
// Test: comb.parity taint rule: y_t = OR-reduce(input_t)
// CHECK-LABEL: hw.module @test_parity
// CHECK-SAME: (in %a : i8, in %a_t : i8, out y : i1, out y_t : i1)
hw.module @test_parity(in %a : i8, out y : i1) {
  // CHECK: comb.parity %a : i8
  // CHECK: comb.icmp ne %a_t
  %0 = comb.parity %a : i8
  hw.output %0 : i1
}

// -----
// Test: comb.reverse taint rule: y_t = reverse(input_t)
// CHECK-LABEL: hw.module @test_reverse
// CHECK-SAME: (in %a : i8, in %a_t : i8, out y : i8, out y_t : i8)
hw.module @test_reverse(in %a : i8, out y : i8) {
  // CHECK: %0 = comb.reverse %a : i8
  // CHECK: [[Y_T:%.+]] = comb.reverse %a_t : i8
  %0 = comb.reverse %a : i8
  hw.output %0 : i8
}

// -----
// Test: seq.compreg taint rule.
// CHECK-LABEL: hw.module @test_compreg
// CHECK-SAME: (in %d : i8, in %d_t : i8, in %clk : !seq.clock, out q : i8, out q_t : i8)
hw.module @test_compreg(in %d : i8, in %clk : !seq.clock, out q : i8) {
  // CHECK: %myreg = seq.compreg %d, %clk : i8
  // CHECK: %myreg_t = seq.compreg sym @myreg_t %d_t, %clk : i8
  %reg = seq.compreg %d, %clk {name = "myreg"} : i8
  hw.output %reg : i8
}

// -----
// Test: seq.compreg with reset - taint register resets to zero.
// CHECK-LABEL: hw.module @test_compreg_reset
// CHECK-SAME: (in %d : i8, in %d_t : i8, in %clk : !seq.clock, in %rst : i1, in %rst_t : i1, out q : i8, out q_t : i8)
hw.module @test_compreg_reset(in %d : i8, in %clk : !seq.clock, in %rst : i1, out q : i8) {
  %c0 = hw.constant 0 : i8
  // CHECK: %myreg = seq.compreg %d, %clk reset %rst, %c0_i8 : i8
  // Taint register resets to zero (known value).
  // CHECK: %myreg_t = seq.compreg sym @myreg_t %d_t, %clk reset %rst, %c0_i8{{.*}} : i8
  %reg = seq.compreg %d, %clk reset %rst, %c0 {name = "myreg"} : i8
  hw.output %reg : i8
}

// -----
// Test: seq.firreg taint rule.
// CHECK-LABEL: hw.module @test_firreg
// CHECK-SAME: (in %d : i8, in %d_t : i8, in %clk : !seq.clock, out q : i8, out q_t : i8)
hw.module @test_firreg(in %d : i8, in %clk : !seq.clock, out q : i8) {
  // CHECK: [[REG:%.+]] = seq.firreg %d clock %clk : i8
  // CHECK: [[TREG:%.+]] = seq.firreg %d_t clock %clk : i8
  %reg = seq.firreg %d clock %clk : i8
  hw.output %reg : i8
}

// -----
// Test: Multi-operation dataflow: the demo from the CellIFT paper.
// module demo(input [7:0] a, b, input s, output [7:0] y);
//   assign y = s ? (a + b) : (a ^ b);
// endmodule
// CHECK-LABEL: hw.module @demo
// CHECK-SAME: (in %a : i8, in %a_t : i8, in %b : i8, in %b_t : i8, in %s : i1, in %s_t : i1, out y : i8, out y_t : i8)
hw.module @demo(in %a : i8, in %b : i8, in %s : i1, out y : i8) {
  %add = comb.add %a, %b : i8
  %xor = comb.xor %a, %b : i8
  %y = comb.mux %s, %add, %xor : i8
  // CHECK: comb.add %a, %b : i8
  // CHECK: comb.xor %a, %b : i8
  // CHECK: comb.mux %s, {{%.+}}, {{%.+}} : i8
  // The output should include both y and y_t.
  // CHECK: hw.output {{%.+}}, {{%.+}} : i8, i8
  hw.output %y : i8
}

// -----
// Test: Hierarchy - child module gets instrumented, parent's instance is updated.
// CHECK-LABEL: hw.module @child
// CHECK-SAME: (in %x : i8, in %x_t : i8, out y : i8, out y_t : i8)
hw.module @child(in %x : i8, out y : i8) {
  %c1 = hw.constant 1 : i8
  %r = comb.add %x, %c1 : i8
  hw.output %r : i8
}

// CHECK-LABEL: hw.module @parent
// CHECK-SAME: (in %a : i8, in %a_t : i8, out z : i8, out z_t : i8)
hw.module @parent(in %a : i8, out z : i8) {
  // The instance should be updated to pass taint ports.
  // CHECK: hw.instance "u" @child(x: %a: i8, x_t: %a_t: i8) -> (y: i8, y_t: i8)
  %r = hw.instance "u" @child(x: %a: i8) -> (y: i8)
  hw.output %r : i8
}

// -----
// Test: Multiple outputs.
// CHECK-LABEL: hw.module @test_multi_output
// CHECK-SAME: (in %a : i8, in %a_t : i8, out x : i8, out x_t : i8, out y : i8, out y_t : i8)
hw.module @test_multi_output(in %a : i8, out x : i8, out y : i8) {
  %c1 = hw.constant 1 : i8
  %r1 = comb.add %a, %c1 : i8
  %r2 = comb.xor %a, %c1 : i8
  hw.output %r1, %r2 : i8, i8
}

// -----
// Test: i1 operations (special case - no need for replicate/OR-reduce).
// CHECK-LABEL: hw.module @test_i1
// CHECK-SAME: (in %a : i1, in %a_t : i1, in %b : i1, in %b_t : i1, out y : i1, out y_t : i1)
hw.module @test_i1(in %a : i1, in %b : i1, out y : i1) {
  %0 = comb.xor %a, %b : i1
  // XOR taint on i1: y_t = a_t | b_t
  // CHECK: comb.or %a_t, %b_t : i1
  hw.output %0 : i1
}

// -----
// Test: Chain of operations - ensure taint flows through correctly.
// CHECK-LABEL: hw.module @test_chain
// CHECK-SAME: (in %a : i8, in %a_t : i8, in %b : i8, in %b_t : i8, out y : i8, out y_t : i8)
hw.module @test_chain(in %a : i8, in %b : i8, out y : i8) {
  %c = comb.xor %a, %b : i8
  %d = comb.and %c, %a : i8
  hw.output %d : i8
}

// -----
// Test: comb.mux with i1 operands (special broadcast case).
// CHECK-LABEL: hw.module @test_mux_i1
// CHECK-SAME: (in %sel : i1, in %sel_t : i1, in %a : i1, in %a_t : i1, in %b : i1, in %b_t : i1, out y : i1, out y_t : i1)
hw.module @test_mux_i1(in %sel : i1, in %a : i1, in %b : i1, out y : i1) {
  %0 = comb.mux %sel, %a, %b : i1
  // For i1, sel_t broadcast is just sel_t itself (no replicate needed).
  hw.output %0 : i1
}

// -----
// Test: Clock port should NOT get a taint port (not integer type).
// CHECK-LABEL: hw.module @test_clock_no_taint
// CHECK-SAME: (in %clk : !seq.clock, in %d : i8, in %d_t : i8, out q : i8, out q_t : i8)
hw.module @test_clock_no_taint(in %clk : !seq.clock, in %d : i8, out q : i8) {
  %reg = seq.compreg %d, %clk {name = "r"} : i8
  hw.output %reg : i8
}

// -----
// Test: Register feeding back to itself (sequential loop).
// CHECK-LABEL: hw.module @test_counter
// CHECK-SAME: (in %clk : !seq.clock, in %en : i1, in %en_t : i1, out count : i8, out count_t : i8)
hw.module @test_counter(in %clk : !seq.clock, in %en : i1, out count : i8) {
  %c1 = hw.constant 1 : i8
  %next = comb.add %reg, %c1 : i8
  %reg = seq.compreg %next, %clk {name = "cnt"} : i8
  hw.output %reg : i8
}

// -----
// Test: comb.icmp ugt precise taint rule (min/max comparison).
// y_t = ugt(min_a, max_b) XOR ugt(max_a, min_b)
// CHECK-LABEL: hw.module @test_icmp_ugt
// CHECK-SAME: (in %a : i8, in %a_t : i8, in %b : i8, in %b_t : i8, out y : i1, out y_t : i1)
hw.module @test_icmp_ugt(in %a : i8, in %b : i8, out y : i1) {
  // CHECK: comb.icmp ugt %a, %b : i8
  // min = val & ~taint, max = val | taint.
  // CHECK: comb.and %a, {{%.+}} : i8
  // CHECK: comb.or %a, %a_t : i8
  // CHECK: comb.and %b, {{%.+}} : i8
  // CHECK: comb.or %b, %b_t : i8
  // Compare extremes.
  // CHECK: [[CMP1:%.+]] = comb.icmp ugt {{%.+}}, {{%.+}} : i8
  // CHECK: [[CMP2:%.+]] = comb.icmp ugt {{%.+}}, {{%.+}} : i8
  // CHECK: [[Y_T:%.+]] = comb.xor [[CMP1]], [[CMP2]] : i1
  %0 = comb.icmp ugt %a, %b : i8
  hw.output %0 : i1
}

// -----
// Test: comb.icmp slt precise taint rule (signed comparison).
// Signed MSB: min -> set to 1 (most negative), max -> clear to 0 (most positive).
// CHECK-LABEL: hw.module @test_icmp_slt
// CHECK-SAME: (in %a : i8, in %a_t : i8, in %b : i8, in %b_t : i8, out y : i1, out y_t : i1)
hw.module @test_icmp_slt(in %a : i8, in %b : i8, out y : i1) {
  // CHECK: comb.icmp slt %a, %b : i8
  // Extract LSBs and MSB for signed handling.
  // CHECK: comb.extract {{%.+}} from 0 : (i8) -> i7
  // CHECK: comb.extract {{%.+}} from 7 : (i8) -> i1
  // Reconstruct min/max with concat.
  // CHECK: comb.concat {{%.+}}, {{%.+}} : i1, i7
  // Compare extremes with slt.
  // CHECK: [[CMP1:%.+]] = comb.icmp slt {{%.+}}, {{%.+}} : i8
  // CHECK: [[CMP2:%.+]] = comb.icmp slt {{%.+}}, {{%.+}} : i8
  // CHECK: [[Y_T:%.+]] = comb.xor [[CMP1]], [[CMP2]] : i1
  %0 = comb.icmp slt %a, %b : i8
  hw.output %0 : i1
}

// -----
// Test: comb.icmp ne precise taint rule (same structure as eq).
// CHECK-LABEL: hw.module @test_icmp_ne
// CHECK-SAME: (in %a : i8, in %a_t : i8, in %b : i8, in %b_t : i8, out y : i1, out y_t : i1)
hw.module @test_icmp_ne(in %a : i8, in %b : i8, out y : i1) {
  // CHECK: comb.icmp ne %a, %b : i8
  // Mask-and-compare approach.
  // CHECK: comb.or %a_t, %b_t : i8
  // CHECK: comb.icmp ne {{%.+}}, {{%.+}} : i8
  // CHECK: comb.icmp eq {{%.+}}, {{%.+}} : i8
  // CHECK: comb.and {{%.+}}, {{%.+}} : i1
  %0 = comb.icmp ne %a, %b : i8
  hw.output %0 : i1
}
