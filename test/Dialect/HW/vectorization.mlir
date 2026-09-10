// RUN: circt-opt %s -hw-vectorization | FileCheck %s

// ---------- Identity permutation ----------

// CHECK-LABEL: hw.module @identity_permutation_vectorization(
// CHECK: hw.output %in : i4
hw.module @identity_permutation_vectorization(in %in : i4, out out : i4) {
  %in_0 = comb.extract %in from 0 : (i4) -> i1
  %in_1 = comb.extract %in from 1 : (i4) -> i1
  %in_2 = comb.extract %in from 2 : (i4) -> i1
  %in_3 = comb.extract %in from 3 : (i4) -> i1
  %fwd = comb.concat %in_3, %in_2, %in_1, %in_0 : i1, i1, i1, i1
  hw.output %fwd : i4
}

// ---------- Reverse endianess: should use comb.reverse ----------

// CHECK-LABEL: hw.module @reverse_endianess_vectorization(
// CHECK: %0 = comb.reverse %in : i4
// CHECK: hw.output %0 : i4
hw.module @reverse_endianess_vectorization(in %in : i4, out out : i4) {
  %in_0 = comb.extract %in from 0 : (i4) -> i1
  %in_1 = comb.extract %in from 1 : (i4) -> i1
  %in_2 = comb.extract %in from 2 : (i4) -> i1
  %in_3 = comb.extract %in from 3 : (i4) -> i1
  %rev = comb.concat %in_0, %in_1, %in_2, %in_3 : i1, i1, i1, i1
  hw.output %rev : i4
}

// ---------- Mixed permutation: bits [3,1:2,0] -> extract+concat chunks ----------
//
// Input concat (MSB->LSB): %in_0, %in_2, %in_1, %in_3
//   out[3]=in[0], out[2]=in[2], out[1]=in[1], out[0]=in[3]
//
// Mix grouping (LSB->MSB):
//   i=0: startBit=3, len=1  -> extract %in from 3 : i1
//   i=1: startBit=1, len=2  -> extract %in from 1 : i2  (in[1] and in[2] are consecutive)
//   i=3: startBit=0, len=1  -> extract %in from 0 : i1
//
// After reversing chunks for concat (MSB->LSB order):
//   concat(extract[0:i1], extract[1:i2], extract[3:i1])

// CHECK-LABEL: hw.module @mixed_vectorization(
// CHECK-DAG: [[C0:%[0-9]+]] = comb.extract %in from 0 : (i4) -> i1
// CHECK-DAG: [[C1:%[0-9]+]] = comb.extract %in from 1 : (i4) -> i2
// CHECK-DAG: [[C3:%[0-9]+]] = comb.extract %in from 3 : (i4) -> i1
// CHECK: comb.concat [[C0]], [[C1]], [[C3]] : i1, i2, i1
hw.module @mixed_vectorization(in %in : i4, out out : i4) {
  %in_0 = comb.extract %in from 0 : (i4) -> i1
  %in_1 = comb.extract %in from 1 : (i4) -> i1
  %in_2 = comb.extract %in from 2 : (i4) -> i1
  %in_3 = comb.extract %in from 3 : (i4) -> i1
  // Permutation (MSB->LSB): out[3]=in[0], out[2]=in[2], out[1]=in[1], out[0]=in[3]
  %mix = comb.concat %in_0, %in_2, %in_1, %in_3 : i1, i1, i1, i1
  hw.output %mix : i4
}

// ---------- Negative Case: Bits from different sources ----------
// CHECK-LABEL: hw.module @multi_source_no_vectorization
// CHECK:         comb.concat
hw.module @multi_source_no_vectorization(in %a : i2, in %b : i2, out out : i4) {
  %a0 = comb.extract %a from 0 : (i2) -> i1
  %a1 = comb.extract %a from 1 : (i2) -> i1
  %b0 = comb.extract %b from 0 : (i2) -> i1
  %b1 = comb.extract %b from 1 : (i2) -> i1
  
  // Mixing pieces of %a and %b. Should not be vectorized to a single source.
  %mix = comb.concat %a1, %b1, %a0, %b0 : i1, i1, i1, i1
  hw.output %mix : i4
}

// ---------- Negative Case: Duplicate bits ----------
// CHECK-LABEL: hw.module @duplicate_bits_no_vectorization
// CHECK:         comb.concat
hw.module @duplicate_bits_no_vectorization(in %in : i4, out out : i4) {
  %in_0 = comb.extract %in from 0 : (i4) -> i1
  %in_1 = comb.extract %in from 1 : (i4) -> i1
  
  // It uses bit 0 and bit 1 twice. It is not a 1:1 permutation.
  %dupe = comb.concat %in_1, %in_1, %in_0, %in_0 : i1, i1, i1, i1
  hw.output %dupe : i4
}

// ---------- Structural XOR Vectorization ----------
// Patterns: 4 independent XOR gates with bits from %a and %b.
// Transformation: Should collapse into a single 4-bit XOR.

// CHECK-LABEL: hw.module @structural_xor(
// CHECK-NEXT:    [[RES:%.+]] = comb.xor %a, %b : i4
// CHECK-NEXT:    hw.output [[RES]] : i4
hw.module @structural_xor(in %a : i4, in %b : i4, out out : i4) {
  %a0 = comb.extract %a from 0 : (i4) -> i1
  %a1 = comb.extract %a from 1 : (i4) -> i1
  %a2 = comb.extract %a from 2 : (i4) -> i1
  %a3 = comb.extract %a from 3 : (i4) -> i1

  %b0 = comb.extract %b from 0 : (i4) -> i1
  %b1 = comb.extract %b from 1 : (i4) -> i1
  %b2 = comb.extract %b from 2 : (i4) -> i1
  %b3 = comb.extract %b from 3 : (i4) -> i1

  %xor0 = comb.xor %a0, %b0 : i1
  %xor1 = comb.xor %a1, %b1 : i1
  %xor2 = comb.xor %a2, %b2 : i1
  %xor3 = comb.xor %a3, %b3 : i1

  %out = comb.concat %xor3, %xor2, %xor1, %xor0 : i1, i1, i1, i1
  hw.output %out : i4
}

// ---------- Structural Mux Vectorization (Shared Control) ----------
// `sel` is a shared i1 BlockArgument; areSubgraphsEquivalent treats it as a
// shared control signal (the !opA && !opB leaf case), so it is passed directly
// to the wide comb.mux without replication.
// CHECK-LABEL: hw.module @structural_mux(
// CHECK-NEXT:    %[[RES:.+]] = comb.mux %sel, %a, %b : i4
// CHECK-NEXT:    hw.output %[[RES]] : i4
hw.module @structural_mux(in %a : i4, in %b : i4, in %sel : i1, out out : i4) {
  %a0 = comb.extract %a from 0 : (i4) -> i1
  %a1 = comb.extract %a from 1 : (i4) -> i1
  %a2 = comb.extract %a from 2 : (i4) -> i1
  %a3 = comb.extract %a from 3 : (i4) -> i1

  %b0 = comb.extract %b from 0 : (i4) -> i1
  %b1 = comb.extract %b from 1 : (i4) -> i1
  %b2 = comb.extract %b from 2 : (i4) -> i1
  %b3 = comb.extract %b from 3 : (i4) -> i1

  %m0 = comb.mux %sel, %a0, %b0 : i1
  %m1 = comb.mux %sel, %a1, %b1 : i1
  %m2 = comb.mux %sel, %a2, %b2 : i1
  %m3 = comb.mux %sel, %a3, %b3 : i1

  %out = comb.concat %m3, %m2, %m1, %m0 : i1, i1, i1, i1
  hw.output %out : i4
}

// ---------- Structural AND with scalar enable ----------
// Models: assign o[i] = a[i] & enable; for all i.
// `enable` is a shared i1 BlockArgument used as a *data* operand (not a
// selector), so vectorizeSubgraph broadcasts it via comb.replicate to match
// the vector width before emitting the wide AND.
//
// CHECK-LABEL: hw.module @structural_and_enable(
// CHECK-NEXT:    %[[REP:.+]] = comb.replicate %enable : (i1) -> i4
// CHECK-NEXT:    %[[RES:.+]] = comb.and %a, %[[REP]] : i4
// CHECK-NEXT:    hw.output %[[RES]] : i4
hw.module @structural_and_enable(in %a : i4, in %enable : i1, out out : i4) {
  %a0 = comb.extract %a from 0 : (i4) -> i1
  %a1 = comb.extract %a from 1 : (i4) -> i1
  %a2 = comb.extract %a from 2 : (i4) -> i1
  %a3 = comb.extract %a from 3 : (i4) -> i1

  %and0 = comb.and %a0, %enable : i1
  %and1 = comb.and %a1, %enable : i1
  %and2 = comb.and %a2, %enable : i1
  %and3 = comb.and %a3, %enable : i1

  %out = comb.concat %and3, %and2, %and1, %and0 : i1, i1, i1, i1
  hw.output %out : i4
}

// ---------- Structural AND with vector enable ----------
// Models: assign o[i] = a[i] & enable[i]; for all i.
// Both operands advance by 1 bit per lane, so the pattern is recognized
// as a standard 2-operand structural AND between two i4 vectors.
//
// CHECK-LABEL: hw.module @structural_and_vector_enable(
// CHECK-NEXT:    [[RES:%.+]] = comb.and %a, %enable : i4
// CHECK-NEXT:    hw.output [[RES]] : i4
hw.module @structural_and_vector_enable(in %a : i4, in %enable : i4, out out : i4) {
  %a0 = comb.extract %a from 0 : (i4) -> i1
  %a1 = comb.extract %a from 1 : (i4) -> i1
  %a2 = comb.extract %a from 2 : (i4) -> i1
  %a3 = comb.extract %a from 3 : (i4) -> i1

  %e0 = comb.extract %enable from 0 : (i4) -> i1
  %e1 = comb.extract %enable from 1 : (i4) -> i1
  %e2 = comb.extract %enable from 2 : (i4) -> i1
  %e3 = comb.extract %enable from 3 : (i4) -> i1

  %and0 = comb.and %a0, %e0 : i1
  %and1 = comb.and %a1, %e1 : i1
  %and2 = comb.and %a2, %e2 : i1
  %and3 = comb.and %a3, %e3 : i1

  %out = comb.concat %and3, %and2, %and1, %and0 : i1, i1, i1, i1
  hw.output %out : i4
}

// ---------- Structural XOR with constant ----------
// Models: assign out[i] = a[i] ^ 1'b1; for all i (bitwise NOT via XOR).
// The constant %c1 is a shared hw.constant with no operands; the recursive
// comparison in areSubgraphsEquivalent returns true for operand-less ops
// with the same name, so all lanes are recognized as isomorphic.
//
// CHECK-LABEL: hw.module @structural_xor_constant(
// CHECK:         [[RES:%.+]] = comb.xor %a
// CHECK-NEXT:    hw.output [[RES]] : i4
hw.module @structural_xor_constant(in %a : i4, out out : i4) {
  %c1 = hw.constant 1 : i1

  %a0 = comb.extract %a from 0 : (i4) -> i1
  %a1 = comb.extract %a from 1 : (i4) -> i1
  %a2 = comb.extract %a from 2 : (i4) -> i1
  %a3 = comb.extract %a from 3 : (i4) -> i1

  %xor0 = comb.xor %a0, %c1 : i1
  %xor1 = comb.xor %a1, %c1 : i1
  %xor2 = comb.xor %a2, %c1 : i1
  %xor3 = comb.xor %a3, %c1 : i1

  %out = comb.concat %xor3, %xor2, %xor1, %xor0 : i1, i1, i1, i1
  hw.output %out : i4
}

// ---------- Multiple output ports (XOR and AND) ----------
// Models test_multiple_patterns from Verilog: two independent output vectors
// sharing the same input, each vectorized independently.
//
// CHECK-LABEL: hw.module @structural_multiple_outputs(
// CHECK-DAG:     [[XOR:%.+]] = comb.xor %a, %b : i4
// CHECK-DAG:     [[AND:%.+]] = comb.and %a, %c : i4
// CHECK:         hw.output [[XOR]], [[AND]] : i4, i4
hw.module @structural_multiple_outputs(
    in %a : i4, in %b : i4, in %c : i4,
    out out_xor : i4, out out_and : i4) {
  %a0 = comb.extract %a from 0 : (i4) -> i1
  %a1 = comb.extract %a from 1 : (i4) -> i1
  %a2 = comb.extract %a from 2 : (i4) -> i1
  %a3 = comb.extract %a from 3 : (i4) -> i1

  %b0 = comb.extract %b from 0 : (i4) -> i1
  %b1 = comb.extract %b from 1 : (i4) -> i1
  %b2 = comb.extract %b from 2 : (i4) -> i1
  %b3 = comb.extract %b from 3 : (i4) -> i1

  %c0 = comb.extract %c from 0 : (i4) -> i1
  %c1 = comb.extract %c from 1 : (i4) -> i1
  %c2 = comb.extract %c from 2 : (i4) -> i1
  %c3 = comb.extract %c from 3 : (i4) -> i1

  %xor0 = comb.xor %a0, %b0 : i1
  %xor1 = comb.xor %a1, %b1 : i1
  %xor2 = comb.xor %a2, %b2 : i1
  %xor3 = comb.xor %a3, %b3 : i1

  %and0 = comb.and %a0, %c0 : i1
  %and1 = comb.and %a1, %c1 : i1
  %and2 = comb.and %a2, %c2 : i1
  %and3 = comb.and %a3, %c3 : i1

  %out_xor = comb.concat %xor3, %xor2, %xor1, %xor0 : i1, i1, i1, i1
  %out_and = comb.concat %and3, %and2, %and1, %and0 : i1, i1, i1, i1
  hw.output %out_xor, %out_and : i4, i4
}

// ---------- Structural OR Vectorization ----------
// Mirrors the XOR test but for OR, ensuring the OrOp lifting path is covered.
//
// CHECK-LABEL: hw.module @structural_or(
// CHECK-NEXT:    [[RES:%.+]] = comb.or %a, %b : i4
// CHECK-NEXT:    hw.output [[RES]] : i4
hw.module @structural_or(in %a : i4, in %b : i4, out out : i4) {
  %a0 = comb.extract %a from 0 : (i4) -> i1
  %a1 = comb.extract %a from 1 : (i4) -> i1
  %a2 = comb.extract %a from 2 : (i4) -> i1
  %a3 = comb.extract %a from 3 : (i4) -> i1

  %b0 = comb.extract %b from 0 : (i4) -> i1
  %b1 = comb.extract %b from 1 : (i4) -> i1
  %b2 = comb.extract %b from 2 : (i4) -> i1
  %b3 = comb.extract %b from 3 : (i4) -> i1

  %or0 = comb.or %a0, %b0 : i1
  %or1 = comb.or %a1, %b1 : i1
  %or2 = comb.or %a2, %b2 : i1
  %or3 = comb.or %a3, %b3 : i1

  %out = comb.concat %or3, %or2, %or1, %or0 : i1, i1, i1, i1
  hw.output %out : i4
}

// ---------- Structural nested operations: (a & b) ^ c ----------
// Each bit lane computes (a[i] & b[i]) ^ c[i]. The recursion in
// areSubgraphsEquivalent must handle depth-2 subgraphs correctly.
//
// CHECK-LABEL: hw.module @structural_nested(
// CHECK:         [[AND:%.+]] = comb.and %a, %b : i4
// CHECK-NEXT:    [[RES:%.+]] = comb.xor [[AND]], %c : i4
// CHECK-NEXT:    hw.output [[RES]] : i4
hw.module @structural_nested(in %a : i4, in %b : i4, in %c : i4, out out : i4) {
  %a0 = comb.extract %a from 0 : (i4) -> i1
  %a1 = comb.extract %a from 1 : (i4) -> i1
  %a2 = comb.extract %a from 2 : (i4) -> i1
  %a3 = comb.extract %a from 3 : (i4) -> i1

  %b0 = comb.extract %b from 0 : (i4) -> i1
  %b1 = comb.extract %b from 1 : (i4) -> i1
  %b2 = comb.extract %b from 2 : (i4) -> i1
  %b3 = comb.extract %b from 3 : (i4) -> i1

  %c0 = comb.extract %c from 0 : (i4) -> i1
  %c1 = comb.extract %c from 1 : (i4) -> i1
  %c2 = comb.extract %c from 2 : (i4) -> i1
  %c3 = comb.extract %c from 3 : (i4) -> i1

  %and0 = comb.and %a0, %b0 : i1
  %and1 = comb.and %a1, %b1 : i1
  %and2 = comb.and %a2, %b2 : i1
  %and3 = comb.and %a3, %b3 : i1

  %xor0 = comb.xor %and0, %c0 : i1
  %xor1 = comb.xor %and1, %c1 : i1
  %xor2 = comb.xor %and2, %c2 : i1
  %xor3 = comb.xor %and3, %c3 : i1

  %out = comb.concat %xor3, %xor2, %xor1, %xor0 : i1, i1, i1, i1
  hw.output %out : i4
}

// ---------- Negative Case: Non-uniform structural pattern ----------
// Bit 0 uses XOR but bits 1-3 use AND — the cones are not isomorphic.
// canVectorizeStructurally must reject this and leave the concat intact.
//
// CHECK-LABEL: hw.module @structural_non_uniform_no_vectorization(
// CHECK:         comb.concat
hw.module @structural_non_uniform_no_vectorization(
    in %a : i4, in %b : i4, out out : i4) {
  %a0 = comb.extract %a from 0 : (i4) -> i1
  %a1 = comb.extract %a from 1 : (i4) -> i1
  %a2 = comb.extract %a from 2 : (i4) -> i1
  %a3 = comb.extract %a from 3 : (i4) -> i1

  %b0 = comb.extract %b from 0 : (i4) -> i1
  %b1 = comb.extract %b from 1 : (i4) -> i1
  %b2 = comb.extract %b from 2 : (i4) -> i1
  %b3 = comb.extract %b from 3 : (i4) -> i1

  // Bit 0 uses XOR, bits 1-3 use AND: not isomorphic, must not vectorize.
  %op0 = comb.xor %a0, %b0 : i1
  %op1 = comb.and %a1, %b1 : i1
  %op2 = comb.and %a2, %b2 : i1
  %op3 = comb.and %a3, %b3 : i1

  %out = comb.concat %op3, %op2, %op1, %op0 : i1, i1, i1, i1
  hw.output %out : i4
}

// ---------- Wider vector (i8) ----------
// Ensures no width is hardcoded anywhere in the pass.
//
// CHECK-LABEL: hw.module @structural_xor_i8(
// CHECK-NEXT:    [[RES:%.+]] = comb.xor %a, %b : i8
// CHECK-NEXT:    hw.output [[RES]] : i8
hw.module @structural_xor_i8(in %a : i8, in %b : i8, out out : i8) {
  %a0 = comb.extract %a from 0 : (i8) -> i1
  %a1 = comb.extract %a from 1 : (i8) -> i1
  %a2 = comb.extract %a from 2 : (i8) -> i1
  %a3 = comb.extract %a from 3 : (i8) -> i1
  %a4 = comb.extract %a from 4 : (i8) -> i1
  %a5 = comb.extract %a from 5 : (i8) -> i1
  %a6 = comb.extract %a from 6 : (i8) -> i1
  %a7 = comb.extract %a from 7 : (i8) -> i1

  %b0 = comb.extract %b from 0 : (i8) -> i1
  %b1 = comb.extract %b from 1 : (i8) -> i1
  %b2 = comb.extract %b from 2 : (i8) -> i1
  %b3 = comb.extract %b from 3 : (i8) -> i1
  %b4 = comb.extract %b from 4 : (i8) -> i1
  %b5 = comb.extract %b from 5 : (i8) -> i1
  %b6 = comb.extract %b from 6 : (i8) -> i1
  %b7 = comb.extract %b from 7 : (i8) -> i1

  %xor0 = comb.xor %a0, %b0 : i1
  %xor1 = comb.xor %a1, %b1 : i1
  %xor2 = comb.xor %a2, %b2 : i1
  %xor3 = comb.xor %a3, %b3 : i1
  %xor4 = comb.xor %a4, %b4 : i1
  %xor5 = comb.xor %a5, %b5 : i1
  %xor6 = comb.xor %a6, %b6 : i1
  %xor7 = comb.xor %a7, %b7 : i1

  %out = comb.concat %xor7, %xor6, %xor5, %xor4, %xor3, %xor2, %xor1, %xor0 : i1, i1, i1, i1, i1, i1, i1, i1
  hw.output %out : i8
}

// ---------- Narrower vector (i2) ----------
// Ensures the pass works correctly at minimal width.
//
// CHECK-LABEL: hw.module @structural_and_i2(
// CHECK-NEXT:    [[RES:%.+]] = comb.and %a, %b : i2
// CHECK-NEXT:    hw.output [[RES]] : i2
hw.module @structural_and_i2(in %a : i2, in %b : i2, out out : i2) {
  %a0 = comb.extract %a from 0 : (i2) -> i1
  %a1 = comb.extract %a from 1 : (i2) -> i1

  %b0 = comb.extract %b from 0 : (i2) -> i1
  %b1 = comb.extract %b from 1 : (i2) -> i1

  %and0 = comb.and %a0, %b0 : i1
  %and1 = comb.and %a1, %b1 : i1

  %out = comb.concat %and1, %and0 : i1, i1
  hw.output %out : i2
}
