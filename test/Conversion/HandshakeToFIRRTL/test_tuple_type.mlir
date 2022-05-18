// RUN: circt-opt -lower-handshake-to-firrtl --split-input-file %s | FileCheck %s

// CHECK-LABEL: firrtl.circuit "test_tuple"  {
// CHECK:  firrtl.module @test_tuple(in %[[VAL_0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: bundle<field0: uint<64>, field1: uint<32>>>, in %[[VAL_1:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_2:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: bundle<field0: uint<64>, field1: uint<32>>>, out %[[VAL_3:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_4:.*]]: !firrtl.clock, in %[[VAL_5:.*]]: !firrtl.uint<1>) {
// CHECK:    firrtl.connect %[[VAL_2]], %[[VAL_0]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: bundle<field0: uint<64>, field1: uint<32>>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: bundle<field0: uint<64>, field1: uint<32>>>
// CHECK:    firrtl.connect %[[VAL_3]], %[[VAL_1]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:  }
// CHECK:}

handshake.func @test_tuple(%arg0: tuple<i64, i32>, %ctrl: none, ...) -> (tuple<i64, i32>, none) {
  return %arg0, %ctrl : tuple<i64, i32>, none
}
