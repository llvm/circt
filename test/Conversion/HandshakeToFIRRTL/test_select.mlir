// RUN: circt-opt -lower-handshake-to-firrtl -split-input-file %s | FileCheck %s

// CHECK-LABEL:   firrtl.circuit "test_select" {
// CHECK:           firrtl.module @arith_select_in_ui1_ui32_ui32_out_ui32(in %[[VAL_0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<1>>, in %[[VAL_1:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in %[[VAL_2:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out %[[VAL_3:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>) {
// CHECK:             %[[VAL_4:.*]] = firrtl.subfield %[[VAL_0]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<1>>
// CHECK:             %[[VAL_5:.*]] = firrtl.subfield %[[VAL_0]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<1>>
// CHECK:             %[[VAL_6:.*]] = firrtl.subfield %[[VAL_0]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<1>>
// CHECK:             %[[VAL_7:.*]] = firrtl.subfield %[[VAL_1]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             %[[VAL_8:.*]] = firrtl.subfield %[[VAL_1]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             %[[VAL_9:.*]] = firrtl.subfield %[[VAL_1]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             %[[VAL_10:.*]] = firrtl.subfield %[[VAL_2]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             %[[VAL_11:.*]] = firrtl.subfield %[[VAL_2]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             %[[VAL_12:.*]] = firrtl.subfield %[[VAL_2]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             %[[VAL_13:.*]] = firrtl.subfield %[[VAL_3]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             %[[VAL_14:.*]] = firrtl.subfield %[[VAL_3]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             %[[VAL_15:.*]] = firrtl.subfield %[[VAL_3]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             %[[VAL_16:.*]] = firrtl.mux(%[[VAL_6]], %[[VAL_9]], %[[VAL_12]]) : (!firrtl.uint<1>, !firrtl.uint<32>, !firrtl.uint<32>) -> !firrtl.uint<32>
// CHECK:             firrtl.connect %[[VAL_15]], %[[VAL_16]] : !firrtl.uint<32>, !firrtl.uint<32>
// CHECK:             %[[VAL_17:.*]] = firrtl.and %[[VAL_7]], %[[VAL_10]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_18:.*]] = firrtl.and %[[VAL_4]], %[[VAL_17]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_19:.*]] = firrtl.and %[[VAL_14]], %[[VAL_18]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_13]], %[[VAL_18]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_5]], %[[VAL_19]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_8]], %[[VAL_19]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_11]], %[[VAL_19]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:           }
// CHECK:           firrtl.module @test_select(in %[[VAL_20:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<1>>, in %[[VAL_21:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in %[[VAL_22:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in %[[VAL_23:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_24:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out %[[VAL_25:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_26:.*]]: !firrtl.clock, in %[[VAL_27:.*]]: !firrtl.uint<1>) {
// CHECK:             %[[VAL_28:.*]], %[[VAL_29:.*]], %[[VAL_30:.*]], %[[VAL_31:.*]] = firrtl.instance arith_select0 @arith_select_in_ui1_ui32_ui32_out_ui32(in in0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<1>>, in in1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in in2: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out out0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>)
// Trivial connects.
// CHECK:           }
// CHECK:         }

handshake.func @test_select(%arg0: i1, %arg1: i32, %arg2: i32, %arg3: none, ...) -> (i32, none) {
  %0 = arith.select %arg0, %arg1, %arg2 : i32
  return %0, %arg3 : i32, none
}
