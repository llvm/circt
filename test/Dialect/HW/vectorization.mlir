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

// ---------- Mixed permutation: should NOT be vectorized ----------
// CHECK-LABEL: hw.module @mixed_no_vectorization
// CHECK:         comb.concat
hw.module @mixed_no_vectorization(in %in : i4, out out : i4) {
  %in_0 = comb.extract %in from 0 : (i4) -> i1
  %in_1 = comb.extract %in from 1 : (i4) -> i1
  %in_2 = comb.extract %in from 2 : (i4) -> i1
  %in_3 = comb.extract %in from 3 : (i4) -> i1
  // Random permutation (0, 2, 1, 3)
  %mix = comb.concat %in_0, %in_2, %in_1, %in_3 : i1, i1, i1, i1
  hw.output %mix : i4
}
