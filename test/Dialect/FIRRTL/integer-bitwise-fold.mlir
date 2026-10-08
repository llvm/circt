// RUN: circt-opt --canonicalize %s | FileCheck %s

firrtl.circuit "IntegerBitwiseFoldCircuit" {
  firrtl.module @IntegerBitwiseFoldCircuit() {}
  firrtl.class @IntegerBitwiseFold(out %outAndEqual: !firrtl.integer,
      out %outOrEqual: !firrtl.integer, out %outAndMixed: !firrtl.integer,
      out %outOrMixed: !firrtl.integer, out %outAndWideNeg: !firrtl.integer,
      out %outOrWideNeg: !firrtl.integer, out %outNotZero: !firrtl.integer,
      out %outNotNegative: !firrtl.integer) {
    %neg2 = firrtl.integer -2
    %one = firrtl.integer 1
    %five = firrtl.integer 5
    %sixteen = firrtl.integer 16
    %negativeSixteen = firrtl.integer -16
    %zero = firrtl.integer 0

    // Equal-width values (-2 and 1 both need two signed bits).
    // CHECK-DAG: [[AND_EQUAL:%.+]] = firrtl.integer 0
    %andEqual = firrtl.integer.and %neg2, %one : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer
    // CHECK-DAG: [[OR_EQUAL:%.+]] = firrtl.integer -1
    %orEqual = firrtl.integer.or %neg2, %one : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer

    // A narrower negative operand must sign-extend. Zero extension would make
    // this AND produce zero instead of 16.
    // CHECK-DAG: [[AND_MIXED:%.+]] = firrtl.integer 16
    %andMixed = firrtl.integer.and %neg2, %sixteen : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer
    // CHECK-DAG: [[OR_MIXED:%.+]] = firrtl.integer -2
    %orMixed = firrtl.integer.or %neg2, %sixteen : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer

    // A narrower positive operand with a wider negative operand.
    %andWideNeg = firrtl.integer.and %negativeSixteen, %five : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer
    %orWideNeg = firrtl.integer.or %negativeSixteen, %five : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer

    // NOT uses arbitrary-precision signed two's-complement semantics.
    %notZero = firrtl.integer.not %zero : (!firrtl.integer) -> !firrtl.integer
    %notNegative = firrtl.integer.not %neg2 : (!firrtl.integer) -> !firrtl.integer

    // CHECK-DAG: [[AND_WIDE_NEG:%.+]] = firrtl.integer 0
    // CHECK-DAG: [[OR_WIDE_NEG:%.+]] = firrtl.integer -11
    // CHECK-DAG: [[NOT_ZERO:%.+]] = firrtl.integer -1
    // CHECK-DAG: [[NOT_NEGATIVE:%.+]] = firrtl.integer 1
    // CHECK: firrtl.propassign %outAndEqual, [[AND_EQUAL]]
    firrtl.propassign %outAndEqual, %andEqual : !firrtl.integer
    // CHECK: firrtl.propassign %outOrEqual, [[OR_EQUAL]]
    firrtl.propassign %outOrEqual, %orEqual : !firrtl.integer
    // CHECK: firrtl.propassign %outAndMixed, [[AND_MIXED]]
    firrtl.propassign %outAndMixed, %andMixed : !firrtl.integer
    // CHECK: firrtl.propassign %outOrMixed, [[OR_MIXED]]
    firrtl.propassign %outOrMixed, %orMixed : !firrtl.integer
    // CHECK: firrtl.propassign %outAndWideNeg, [[AND_WIDE_NEG]]
    firrtl.propassign %outAndWideNeg, %andWideNeg : !firrtl.integer
    // CHECK: firrtl.propassign %outOrWideNeg, [[OR_WIDE_NEG]]
    firrtl.propassign %outOrWideNeg, %orWideNeg : !firrtl.integer
    // CHECK: firrtl.propassign %outNotZero, [[NOT_ZERO]]
    firrtl.propassign %outNotZero, %notZero : !firrtl.integer
    // CHECK: firrtl.propassign %outNotNegative, [[NOT_NEGATIVE]]
    firrtl.propassign %outNotNegative, %notNegative : !firrtl.integer
  }
}
