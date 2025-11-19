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

// ---------- Mixed permutation ----------

// CHECK-LABEL: hw.module @mix_simple(
// CHECK: hw.constant
// CHECK: comb.concat
// CHECK: hw.output
hw.module @mix_simple(in %in : i4, out out : i4) {
  %c1_i1 = hw.constant 1 : i1
  %in_2 = comb.extract %in from 2 : (i4) -> i1
  %in_1 = comb.extract %in from 1 : (i4) -> i1
  %in_0 = comb.extract %in from 0 : (i4) -> i1
  %mix = comb.concat %c1_i1, %in_2, %in_1, %in_0 : i1, i1, i1, i1
  hw.output %mix : i4
}

// ---------- Complex pattern(MUX): should vectorize into replicated select logic ---------- 

// CHECK-LABEL: hw.module @test_mux(
// CHECK: %true = hw.constant true
// CHECK: %0 = comb.replicate %sel : (i1) -> i4
// CHECK: %1 = comb.and %a, %0 : i4
// CHECK: %2 = comb.replicate %sel : (i1) -> i4
// CHECK: %3 = comb.replicate %true : (i1) -> i4
// CHECK: %4 = comb.xor %2, %3 : i4
// CHECK: %5 = comb.and %b, %4 : i4
// CHECK: %6 = comb.or %1, %5 : i4
// CHECK: hw.output %6 : i4
hw.module @test_mux(in %a : i4, in %b : i4, in %sel : i1, out result : i4) {
  %c0_i2 = hw.constant 0 : i2
  %false = hw.constant false
  %c7_i4 = hw.constant 7 : i4
  %c-5_i4 = hw.constant -5 : i4
  %true = hw.constant true
  %c0_i3 = hw.constant 0 : i3
  %0 = comb.concat %c0_i3, %29 : i3, i1
  %1 = comb.concat %c0_i2, %24, %false : i2, i1, i1
  %2 = comb.or %1, %0 : i4
  %3 = comb.and %2, %c-5_i4 : i4
  %4 = comb.concat %false, %19, %c0_i2 : i1, i1, i2
  %5 = comb.or %4, %3 : i4
  %6 = comb.and %5, %c7_i4 : i4
  %7 = comb.concat %14, %c0_i3 : i1, i3
  %8 = comb.or %7, %6 : i4
  %9 = comb.extract %a from 3 : (i4) -> i1
  %10 = comb.and %9, %sel : i1
  %11 = comb.extract %b from 3 : (i4) -> i1
  %12 = comb.xor %sel, %true : i1
  %13 = comb.and %11, %12 : i1
  %14 = comb.or %10, %13 : i1
  %15 = comb.extract %a from 2 : (i4) -> i1
  %16 = comb.and %15, %sel : i1
  %17 = comb.extract %b from 2 : (i4) -> i1
  %18 = comb.and %17, %12 : i1
  %19 = comb.or %16, %18 : i1
  %20 = comb.extract %a from 1 : (i4) -> i1
  %21 = comb.and %20, %sel : i1
  %22 = comb.extract %b from 1 : (i4) -> i1
  %23 = comb.and %22, %12 : i1
  %24 = comb.or %21, %23 : i1
  %25 = comb.extract %a from 0 : (i4) -> i1
  %26 = comb.and %25, %sel : i1
  %27 = comb.extract %b from 0 : (i4) -> i1
  %28 = comb.and %27, %12 : i1
  %29 = comb.or %26, %28 : i1
  hw.output %8 : i4
}

// ---------- Non-vectorizable pattern (cross dependencies) ----------

// CHECK-LABEL: hw.module @cross_dependency(
// CHECK: comb.extract
// CHECK: comb.xor
// CHECK: comb.concat
// CHECK: hw.output
hw.module @cross_dependency(in %in : i2, out out : i2) {
  // Aqui criamos dependências cruzadas entre os bits
  %0 = comb.extract %in from 0 : (i2) -> i1
  %1 = comb.extract %6 from 1 : (i2) -> i1
  %2 = comb.xor %0, %1 : i1
  %3 = comb.extract %in from 1 : (i2) -> i1
  %4 = comb.extract %6 from 0 : (i2) -> i1
  %5 = comb.xor %3, %4 : i1
  %6 = comb.concat %5, %2 : i1, i1
  hw.output %6 : i2
}
