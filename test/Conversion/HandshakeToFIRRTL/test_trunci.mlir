// RUN: circt-opt -lower-handshake-to-firrtl -split-input-file %s | FileCheck %s

// CHECK-LABEL:     firrtl.module @arith_trunci_in_ui32_out_ui8(
// CHECK-SAME:          in %[[VAL_0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>,
// CHECK-SAME:          out %[[VAL_1:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>) {
// CHECK:             %[[VAL_2:.*]] = firrtl.subfield %[[VAL_0]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             %[[VAL_3:.*]] = firrtl.subfield %[[VAL_0]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             %[[VAL_4:.*]] = firrtl.subfield %[[VAL_0]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             %[[VAL_5:.*]] = firrtl.subfield %[[VAL_1]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>
// CHECK:             %[[VAL_6:.*]] = firrtl.subfield %[[VAL_1]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>
// CHECK:             %[[VAL_7:.*]] = firrtl.subfield %[[VAL_1]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<8>>
// CHECK:             %[[VAL_8:.*]] = firrtl.bits %[[VAL_4]] 7 to 0 : (!firrtl.uint<32>) -> !firrtl.uint<8>
// CHECK:             firrtl.connect %[[VAL_7]], %[[VAL_8]] : !firrtl.uint<8>, !firrtl.uint<8>
// CHECK:             firrtl.connect %[[VAL_5]], %[[VAL_2]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_9:.*]] = firrtl.and %[[VAL_6]], %[[VAL_2]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_3]], %[[VAL_9]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:           }

handshake.func @test_trunci(%arg0: i32, ...) -> i8 {
  %res = arith.trunci %arg0 : i32 to i8
  return %res : i8
}
