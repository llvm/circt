// RUN: circt-opt --cellift-instrument --split-input-file %s | FileCheck %s
//
// This file verifies that the CIRCT CellIFT implementation produces equivalent
// taint rules to the Yosys CellIFT implementation (comsec-group/cellift-yosys)
// for operations that currently have precise CIRCT models.
//
// Operations that currently use conservative implementation, such as shifts and
// modulo, only check the shape of the conservative instrumentation.

// -----
// Equivalence: ADD precise taint.
// Yosys add.cc: y_t = ((a & ~a_t) + (b & ~b_t)) XOR ((a | a_t) + (b | b_t)) | a_t | b_t
// The min/max XOR approach tracks carry chain taint precisely.
// CHECK-LABEL: hw.module @equiv_add
hw.module @equiv_add(in %a : i8, in %b : i8, out y : i8) {
  // CHECK: comb.add %a, %b : i8
  // CHECK: comb.xor %a_t, %c-1_i8
  // CHECK: comb.xor %b_t, %c-1_i8
  // CHECK: [[AZERO:%.+]] = comb.and %a
  // CHECK: [[BZERO:%.+]] = comb.and %b
  // CHECK: [[AONE:%.+]] = comb.or %a, %a_t
  // CHECK: [[BONE:%.+]] = comb.or %b, %b_t
  // CHECK: [[SMIN:%.+]] = comb.add [[AZERO]], [[BZERO]]
  // CHECK: [[SMAX:%.+]] = comb.add [[AONE]], [[BONE]]
  // CHECK: [[XOR:%.+]] = comb.xor [[SMIN]], [[SMAX]]
  // CHECK: comb.or [[XOR]], %a_t, %b_t
  %0 = comb.add %a, %b : i8
  hw.output %0 : i8
}

// -----
// Equivalence: SUB precise taint.
// Yosys sub.cc: y_t = ((a | a_t) - (b & ~b_t)) XOR ((a & ~a_t) - (b | b_t)) | a_t | b_t
// CHECK-LABEL: hw.module @equiv_sub
hw.module @equiv_sub(in %a : i8, in %b : i8, out y : i8) {
  // CHECK: comb.sub %a, %b : i8
  // CHECK: [[AONE:%.+]] = comb.or %a, %a_t
  // CHECK: [[AZERO:%.+]] = comb.and %a
  // CHECK: [[S1:%.+]] = comb.sub [[AONE]]
  // CHECK: [[S2:%.+]] = comb.sub [[AZERO]]
  // CHECK: [[XOR:%.+]] = comb.xor [[S1]], [[S2]]
  // CHECK: comb.or [[XOR]], %a_t, %b_t
  %0 = comb.sub %a, %b : i8
  hw.output %0 : i8
}

// -----
// Equivalence: MUX taint.
// Yosys mux.cc: data_taint = (a_t & (~s|s_t)) | (b_t & (s|s_t))
//               ctrl_taint = s_t & (a ^ b)
//               y_t = data_taint | ctrl_taint
// CIRCT: y_t = mux(sel, t_t, f_t) | replicate(sel_t) & (t^f | t_t | f_t)
// These are algebraically equivalent (verified).
// CHECK-LABEL: hw.module @equiv_mux
hw.module @equiv_mux(in %sel : i1, in %a : i8, in %b : i8, out y : i8) {
  // CHECK: comb.mux %sel, %a, %b : i8
  // CHECK: comb.mux %sel, %a_t, %b_t : i8
  // CHECK: comb.replicate %sel_t
  // CHECK: comb.xor %a, %b : i8
  // CHECK: comb.or {{%.+}}, %a_t, %b_t : i8
  // CHECK: comb.and
  // CHECK: comb.or
  %0 = comb.mux %sel, %a, %b : i8
  hw.output %0 : i8
}

// -----
// Equivalence: EQ precise taint.
// Yosys eq_ne.cc: has_taint = reduce_or(a_t | b_t)
//                 mask = ~(a_t | b_t)
//                 y_t = has_taint & ((a & mask) == (b & mask))
// CHECK-LABEL: hw.module @equiv_eq
hw.module @equiv_eq(in %a : i8, in %b : i8, out y : i1) {
  // CHECK: comb.icmp eq %a, %b : i8
  // CHECK: [[COMB:%.+]] = comb.or %a_t, %b_t : i8
  // CHECK: [[HAS:%.+]] = comb.icmp ne [[COMB]]
  // CHECK: [[MASK:%.+]] = comb.xor [[COMB]], %c-1_i8 : i8
  // CHECK: [[MA:%.+]] = comb.and %a, [[MASK]]
  // CHECK: [[MB:%.+]] = comb.and %b, [[MASK]]
  // CHECK: [[EQ:%.+]] = comb.icmp eq [[MA]], [[MB]]
  // CHECK: comb.and [[HAS]], [[EQ]]
  %0 = comb.icmp eq %a, %b : i8
  hw.output %0 : i1
}

// -----
// Equivalence: GE unsigned precise taint.
// Yosys ge.cc: min_a = a & ~a_t, max_a = a | a_t (all bits)
//              y_t = ge(min_a, max_b) XOR ge(max_a, min_b)
// CHECK-LABEL: hw.module @equiv_uge
hw.module @equiv_uge(in %a : i8, in %b : i8, out y : i1) {
  // CHECK: comb.icmp uge %a, %b : i8
  // CHECK: [[MINA:%.+]] = comb.and %a
  // CHECK: [[MAXA:%.+]] = comb.or %a, %a_t
  // CHECK: [[MINB:%.+]] = comb.and %b
  // CHECK: [[MAXB:%.+]] = comb.or %b, %b_t
  // CHECK: [[C1:%.+]] = comb.icmp uge [[MINA]], [[MAXB]]
  // CHECK: [[C2:%.+]] = comb.icmp uge [[MAXA]], [[MINB]]
  // CHECK: comb.xor [[C1]], [[C2]]
  %0 = comb.icmp uge %a, %b : i8
  hw.output %0 : i1
}

// -----
// Equivalence: SGE signed precise taint.
// Yosys ge.cc (signed): MSB min->1 (negative), max->0 (positive).
//                        LSBs: min->0, max->1.
// CHECK-LABEL: hw.module @equiv_sge
hw.module @equiv_sge(in %a : i8, in %b : i8, out y : i1) {
  // CHECK: comb.icmp sge %a, %b : i8
  // Extract LSBs and MSB.
  // CHECK: comb.extract {{%.+}} from 0 : (i8) -> i7
  // CHECK: comb.extract {{%.+}} from 7 : (i8) -> i1
  // MSB: min = a_msb | a_t_msb (set to 1 for negative = minimum).
  // CHECK: comb.or {{%.+}} : i1
  // MSB: max = a_msb & ~a_t_msb (clear to 0 for positive = maximum).
  // CHECK: comb.and {{%.+}} : i1
  // Reconstruct with concat.
  // CHECK: comb.concat {{%.+}}, {{%.+}} : i1, i7
  // Compare extremes.
  // CHECK: [[C1:%.+]] = comb.icmp sge {{%.+}}, {{%.+}} : i8
  // CHECK: [[C2:%.+]] = comb.icmp sge {{%.+}}, {{%.+}} : i8
  // CHECK: comb.xor [[C1]], [[C2]]
  %0 = comb.icmp sge %a, %b : i8
  hw.output %0 : i1
}

// -----
// Coverage: SHL currently uses the conservative shift fallback instead of the
// precise Yosys shl_sshl_precise.cc rule.
// CHECK-LABEL: hw.module @equiv_shl
hw.module @equiv_shl(in %a : i8, in %b : i8, out y : i8) {
  // CHECK: comb.shl %a, %b : i8
  // CHECK: [[HAS_BT:%.+]] = comb.icmp ne %b_t, %c0_i8 : i8
  // CHECK: [[AMT_T:%.+]] = comb.replicate [[HAS_BT]] : (i1) -> i8
  // CHECK: [[SHIFTED_T:%.+]] = comb.shl %a_t, %b : i8
  // CHECK: [[Y_T:%.+]] = comb.or [[AMT_T]], [[SHIFTED_T]] : i8
  %0 = comb.shl %a, %b : i8
  hw.output %0 : i8
}

// -----
// Equivalence: AND taint (GLIFT).
// Yosys and.cc: y_t = (a & b_t) | (b & a_t) | (a_t & b_t)
// CHECK-LABEL: hw.module @equiv_and
hw.module @equiv_and(in %a : i8, in %b : i8, out y : i8) {
  // CHECK: comb.and %a, %b : i8
  // CHECK: [[T1:%.+]] = comb.and %a, %b_t
  // CHECK: [[T2:%.+]] = comb.and %b, %a_t
  // CHECK: [[T3:%.+]] = comb.and %a_t, %b_t
  // CHECK: comb.or [[T1]], [[T2]], [[T3]]
  %0 = comb.and %a, %b : i8
  hw.output %0 : i8
}

// -----
// Equivalence: OR taint (GLIFT).
// Yosys or.cc: y_t = (~a & b_t) | (~b & a_t) | (a_t & b_t)
// CHECK-LABEL: hw.module @equiv_or
hw.module @equiv_or(in %a : i8, in %b : i8, out y : i8) {
  // CHECK: comb.or %a, %b : i8
  // CHECK: comb.xor %a, %c-1_i8
  // CHECK: comb.xor %b, %c-1_i8
  // CHECK: comb.or
  %0 = comb.or %a, %b : i8
  hw.output %0 : i8
}

// -----
// Equivalence: XOR taint.
// Yosys xor.cc: y_t = a_t | b_t
// CHECK-LABEL: hw.module @equiv_xor
hw.module @equiv_xor(in %a : i8, in %b : i8, out y : i8) {
  // CHECK: comb.xor %a, %b : i8
  // CHECK: comb.or %a_t, %b_t
  %0 = comb.xor %a, %b : i8
  hw.output %0 : i8
}

// -----
// Equivalence: MUL taint (conservative, matches Yosys).
// Yosys mul.cc: y_t = replicate(reduce_or(a_t) | reduce_or(b_t))
// CHECK-LABEL: hw.module @equiv_mul
hw.module @equiv_mul(in %a : i8, in %b : i8, out y : i8) {
  // CHECK: comb.mul %a, %b : i8
  // CHECK: comb.icmp ne %a_t
  // CHECK: comb.icmp ne %b_t
  // CHECK: comb.or
  // CHECK: comb.replicate
  %0 = comb.mul %a, %b : i8
  hw.output %0 : i8
}
