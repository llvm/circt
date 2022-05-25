// RUN: circt-opt %s --hw-flatten-structs | FileCheck %s

// CHECK: hw.module @mymod1(%[[VAL_0:.*]]: i8, %[[VAL_1:.*]]: i16) -> (out_c: i16, out_d: i8) {
hw.module @mymod1(%in: !hw.struct<a: i8, b: i16>) -> (out: !hw.struct<c: i16, d: i8>) {
  %0 = hw.struct_extract %in["a"] : !hw.struct<a: i8, b: i16>
  %1 = hw.struct_extract %in["b"] : !hw.struct<a: i8, b: i16>
  %2 = hw.struct_create (%1, %0) : !hw.struct<c: i16, d: i8>
  // CHECK: hw.output %[[VAL_1]], %[[VAL_0]] : i16, i8
  hw.output %2 : !hw.struct<c: i16, d: i8>
}

// CHECK: hw.module @mymod2(%[[VAL_2:.*]]: i8, %[[VAL_3:.*]]: i16) -> (out_c: i16, out_d: i8) {
hw.module @mymod2(%in: !hw.struct<a: i8, b: i16>) -> (out: !hw.struct<c: i16, d: i8>) {
  // CHECK: %[[VAL_4:.*]], %[[VAL_5:.*]] = hw.instance "myinst" sym @myinst @mymod1(in_a: %[[VAL_2]]: i8, in_b: %[[VAL_3]]: i16) -> (out_c: i16, out_d: i8)
  %0 = hw.instance "myinst" @mymod1(in: %in: !hw.struct<a: i8, b: i16>) -> (out: !hw.struct<c: i16, d: i8>)
  // CHECK: hw.output %[[VAL_4]], %[[VAL_5]] : i16, i8
  hw.output %0 : !hw.struct<c: i16, d: i8>
}
