// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s

// CHECK:           firrtl.module @handshake_source_0ins_1outs_ctrl(out %[[VAL_0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>) {
// CHECK:             %[[VAL_1:.*]] = firrtl.subfield %[[VAL_0]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:             %[[VAL_2:.*]] = firrtl.constant 1 : !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_1]], %[[VAL_2]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:           }
// CHECK:           firrtl.module @simple_addi(in %[[VAL_11:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_12:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out %[[VAL_13:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_14:.*]]: !firrtl.clock, in %[[VAL_15:.*]]: !firrtl.uint<1>) {
// CHECK:             %[[VAL_16:.*]] = firrtl.instance handshake_source0  @handshake_source_0ins_1outs_ctrl(out out0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>)
handshake.func @simple_addi(%arg0 : none) -> (i32, none) {
  %0 = source
  %1 = constant %0 {value = 1 : i32} : i32
  return %1, %arg0 : i32, none
}
