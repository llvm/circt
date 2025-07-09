// RUN: circt-opt %s --canonicalize | FileCheck %s

// This test checks two things for comb.reverse:
// 1. Constant folding: reversing a constant should be done at compile time.
// 2. Canonicalization: the pattern reverse(reverse(x)) should be simplified to x.

// CHECK-LABEL: hw.module @test_reverse_canonicalize
// CHECK-SAME: (in %[[IN:.*]] : i8, out out_double_rev : i8, out out_const_rev : i8)

// After canonicalization, there should be NO comb.reverse operations,
// since reverse(reverse(x)) simplifies to x, and reverse(const) should be folded.
// Therefore, we ensure there are no comb.reverse ops in the output:
// CHECK-NOT: comb.reverse

hw.module @test_reverse_canonicalize(in %in : i8, out out_double_rev : i8, out out_const_rev : i8) {

  // Apply reverse twice to test canonicalization
  %rev1 = comb.reverse %in : (i8) -> i8
  %rev2 = comb.reverse %rev1 : (i8) -> i8

  // Apply reverse on a constant to test folding
  %c13 = hw.constant 13 : i8
  %rev_const = comb.reverse %c13 : (i8) -> i8

  // Check outputs: after canonicalization,
  // out_double_rev should be wired directly to %in,
  // and out_const_rev should be wired to the folded constant (which is -80 in i8)
  hw.output %rev2, %rev_const : i8, i8
}

// CHECK: %[[CST:.*]] = hw.constant -80 : i8
// CHECK: hw.output %[[IN]], %[[CST]] : i8, i8
