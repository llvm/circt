// REQUIRES: libz3
// REQUIRES: circt-lec-jit

// RUN: circt-opt %s --hw-aggregate-to-comb --convert-comb-to-synth -o %t.mlir

// RUN: circt-lec %t.mlir %s -c1=array -c2=array --shared-libs=%libz3 | FileCheck %s --check-prefix=COMB_ARRAY
// COMB_ARRAY: c1 == c2
hw.module @array(in %arg0: i2, in %arg1: i2, in %arg2: i2, in %arg3: i2, in %sel1: i2, in %sel2: i2, in %sel3: i2, in %val: i2, out out1: i2, out out2: i2) {
  %0 = hw.array_create %arg0, %arg1, %arg2, %arg3 : i2
  %1 = hw.array_get %0[%sel1] : !hw.array<4xi2>, i2
  %2 = hw.array_create %arg0, %arg1, %arg2 : i2
  %c3_i2 = hw.constant 3 : i2
  // NOTE: If the index is out of bounds, the result value is undefined.
  // In LEC such value is lowered into unbounded SMT variable and cause
  // the LEC to fail. So just asssume that the index is in bounds.
  %inbound_sel_2 = comb.icmp ult %sel2, %c3_i2 : i2
  verif.assume %inbound_sel_2 : i1
  %inbound_sel_3 = comb.icmp ult %sel3, %c3_i2 : i2
  verif.assume %inbound_sel_3 : i1
  %inject = hw.array_inject %2[%sel3], %val: !hw.array<3xi2>, i2
  %3 = hw.array_get %inject[%sel2] : !hw.array<3xi2>, i2
  hw.output %1, %3 : i2, i2
}
