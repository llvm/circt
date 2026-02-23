// RUN: circt-opt %s -hw-vectorization | FileCheck %s

// ---------- Identity permutation ----------

// CHECK-LABEL: hw.module @simple_vectorization(
// CHECK: hw.output %in : i4
hw.module @simple_vectorization(in %in : i4, out out : i4) {
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
