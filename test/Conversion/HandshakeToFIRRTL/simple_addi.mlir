// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s

// CHECK: firrtl.module @arith_addi_in_ui64_ui64_out_ui64(in %[[VAL_0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_1:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %[[VAL_2:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) {
// CHECK:   %[[VAL_3:.*]] = firrtl.subfield %[[VAL_0]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:   %[[VAL_4:.*]] = firrtl.subfield %[[VAL_0]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:   %[[VAL_5:.*]] = firrtl.subfield %[[VAL_0]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:   %[[VAL_6:.*]] = firrtl.subfield %[[VAL_1]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:   %[[VAL_7:.*]] = firrtl.subfield %[[VAL_1]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:   %[[VAL_8:.*]] = firrtl.subfield %[[VAL_1]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:   %[[VAL_9:.*]] = firrtl.subfield %[[VAL_2]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:   %[[VAL_10:.*]] = firrtl.subfield %[[VAL_2]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:   %[[VAL_11:.*]] = firrtl.subfield %[[VAL_2]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:   %[[VAL_12:.*]] = firrtl.add %[[VAL_5]], %[[VAL_8]] : (!firrtl.uint<64>, !firrtl.uint<64>) -> !firrtl.uint<65>
// CHECK:   %[[VAL_13:.*]] = firrtl.bits %[[VAL_12]] 63 to 0 : (!firrtl.uint<65>) -> !firrtl.uint<64>
// CHECK:   firrtl.connect %[[VAL_11]], %[[VAL_13]] : !firrtl.uint<64>, !firrtl.uint<64>
// CHECK:   %[[VAL_14:.*]] = firrtl.and %[[VAL_3]], %[[VAL_6]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %[[VAL_9]], %[[VAL_14]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:   %[[VAL_15:.*]] = firrtl.and %[[VAL_10]], %[[VAL_14]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %[[VAL_4]], %[[VAL_15]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:   firrtl.connect %[[VAL_7]], %[[VAL_15]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK: }
// CHECK: firrtl.module @simple_addi(in %[[VAL_16:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_17:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_18:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_19:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %[[VAL_20:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_21:.*]]: !firrtl.clock, in %[[VAL_22:.*]]: !firrtl.uint<1>) {
// CHECK:   %[[VAL_23:.*]], %[[VAL_24:.*]], %[[VAL_25:.*]] = firrtl.instance arith_addi0  @arith_addi_in_ui64_ui64_out_ui64(in [[ARG0:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in [[ARG1:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out [[ARG2:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>)
// CHECK:   firrtl.connect %[[VAL_23]], %[[VAL_16]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:   firrtl.connect %[[VAL_24]], %[[VAL_17]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:   firrtl.connect %[[VAL_19]], %[[VAL_25]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:   firrtl.connect %[[VAL_20]], %[[VAL_18]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK: }
handshake.func @simple_addi(%arg0: index, %arg1: index, %arg2: none, ...) -> (index, none) {
  %0 = arith.addi %arg0, %arg1 : index
  return %0, %arg2 : index, none
}
