// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s

// Submodule for the index and i64 ConstantOps as they have the same value and converted type.
// CHECK-LABEL: firrtl.module @handshake_constant_c42_out_ui64(
// CHECK-SAME: in %[[ARG0:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[ARG1:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) {
// CHECK:   %[[ARG0_VALID:.+]] = firrtl.subfield %[[ARG0]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:   %[[ARG0_READY:.+]] = firrtl.subfield %[[ARG0]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:   %[[ARG1_VALID:.+]] = firrtl.subfield %[[ARG1]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:   %[[ARG1_READY:.+]] = firrtl.subfield %[[ARG1]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:   %[[ARG1_DATA:.+]] = firrtl.subfield %[[ARG1]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:   firrtl.connect %[[ARG1_VALID:.+]], %[[ARG0_VALID:.+]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:   firrtl.connect %[[ARG0_READY:.+]], %[[ARG1_READY:.+]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:   %c42_ui64 = firrtl.constant 42 : !firrtl.uint<64>
// CHECK:   firrtl.connect %[[ARG1_DATA:.+]], %c42_ui64 : !firrtl.uint<64>, !firrtl.uint<64>
// CHECK: }

// Submodule for the ui32 ConstantOp.
// CHECK-LABEL: firrtl.module @handshake_constant_c42_out_ui32(
// CHECK-SAME:  in %[[ARG0:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[ARG1:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>) {
// CHECK:   %c42_ui32 = firrtl.constant 42 : !firrtl.uint<32>

// Submodule for the si32 ConstantOp.
// CHECK-LABEL: firrtl.module @"handshake_constant_c-11_out_si32"(
// CHECK-SAME:  in %[[ARG0:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[ARG1:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: sint<32>>) {
// CHECK:   %c-11_si32 = firrtl.constant -11 : !firrtl.sint<32>

// CHECK: firrtl.module @test_constant(in %[[VAL_97:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_98:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %[[VAL_99:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %[[VAL_100:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out %[[VAL_101:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: sint<32>>, out %[[VAL_102:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_103:.*]]: !firrtl.clock, in %[[VAL_104:.*]]: !firrtl.uint<1>) {
// CHECK:   %[[VAL_105:.*]], %[[VAL_106:.*]], %[[VAL_107:.*]], %[[VAL_108:.*]], %[[VAL_109:.*]], %[[VAL_110:.*]], %[[VAL_111:.*]], %[[VAL_112:.*]] = firrtl.instance handshake_fork0  @handshake_fork_1ins_5outs_ctrl(in [[ARG0:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out [[ARG1:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out [[ARG2:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out [[ARG3:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out [[ARG4:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out [[ARG5:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in clock: !firrtl.clock, in reset: !firrtl.uint<1>)
// CHECK:   %[[VAL_113:.*]], %[[VAL_114:.*]] = firrtl.instance handshake_constant0  @handshake_constant_c42_out_ui64(in [[ARG0:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out [[ARG1:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>)
// CHECK:   %[[VAL_115:.*]], %[[VAL_116:.*]] = firrtl.instance handshake_constant1  @handshake_constant_c42_out_ui64(in [[ARG0:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out [[ARG1:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>)
// CHECK:   %[[VAL_117:.*]], %[[VAL_118:.*]] = firrtl.instance handshake_constant2  @handshake_constant_c42_out_ui32(in [[ARG0:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out [[ARG1:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>)
// CHECK:   %[[VAL_119:.*]], %[[VAL_120:.*]] = firrtl.instance handshake_constant3  @"handshake_constant_c-11_out_si32"(in [[ARG0:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out [[ARG1:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: sint<32>>)
handshake.func @test_constant(%arg0: none, ...) -> (index, i64, ui32, si32, none) {
  %0:5 = fork [5] %arg0 : none
  %1 = constant %0#0 {value = 42 : index} : index
  %2 = constant %0#1 {value = 42 : i64} : i64
  %3 = constant %0#2 {value = 42 : ui32} : ui32
  %4 = constant %0#3 {value = -11 : si32} : si32
  return %1, %2, %3, %4, %0#4 : index, i64, ui32, si32, none
}
