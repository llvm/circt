// RUN: circt-opt -lower-handshake-to-firrtl --split-input-file %s | FileCheck %s

// CHECK-LABEL: firrtl.circuit "test_tuple"  {
// CHECK:  firrtl.module @test_tuple(in %[[VAL_0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: bundle<field0: uint<64>, field1: uint<32>>>, in %[[VAL_1:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_2:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: bundle<field0: uint<64>, field1: uint<32>>>, out %[[VAL_3:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_4:.*]]: !firrtl.clock, in %[[VAL_5:.*]]: !firrtl.uint<1>) {
// CHECK:      %0 = firrtl.subfield %out0[valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: bundle<field0: uint<64>, field1: uint<32>>>
// CHECK:      %1 = firrtl.subfield %arg0[valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: bundle<field0: uint<64>, field1: uint<32>>>
// CHECK:      %2 = firrtl.subfield %out0[ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: bundle<field0: uint<64>, field1: uint<32>>>
// CHECK:      %3 = firrtl.subfield %arg0[ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: bundle<field0: uint<64>, field1: uint<32>>>
// CHECK:      %4 = firrtl.subfield %out0[data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: bundle<field0: uint<64>, field1: uint<32>>>
// CHECK:      %5 = firrtl.subfield %arg0[data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: bundle<field0: uint<64>, field1: uint<32>>>
// CHECK:      %6 = firrtl.subfield %out1[valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:      %7 = firrtl.subfield %ctrl[valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:      %8 = firrtl.subfield %out1[ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:      %9 = firrtl.subfield %ctrl[ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:      firrtl.strictconnect %0, %1 : !firrtl.uint<1>
// CHECK:      firrtl.strictconnect %3, %2 : !firrtl.uint<1>
// CHECK:      firrtl.strictconnect %4, %5 : !firrtl.bundle<field0: uint<64>, field1: uint<32>>
// CHECK:      firrtl.strictconnect %6, %7 : !firrtl.uint<1>
// CHECK:      firrtl.strictconnect %9, %8 : !firrtl.uint<1>
// CHECK:  }
// CHECK:}

handshake.func @test_tuple(%arg0: tuple<i64, i32>, %ctrl: none, ...) -> (tuple<i64, i32>, none) {
  return %arg0, %ctrl : tuple<i64, i32>, none
}
