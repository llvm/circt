// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s

// Submodule for the index and i64 ConstantOps as they have the same value and converted type.
// CHECK-LABEL: firrtl.module @handshake_constant_1ins_1outs_c42_ui64(
// CHECK-SAME: in %arg0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %arg1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) {
// CHECK:   %[[ARG0_VALID:.+]] = firrtl.subfield %arg0(0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>>) -> !firrtl.uint<1>
// CHECK:   %[[ARG0_READY:.+]] = firrtl.subfield %arg0(1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>>) -> !firrtl.uint<1>
// CHECK:   %[[ARG1_VALID:.+]] = firrtl.subfield %arg1(0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %[[ARG1_READY:.+]] = firrtl.subfield %arg1(1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %[[ARG1_DATA:.+]] = firrtl.subfield %arg1(2) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   firrtl.connect %[[ARG1_VALID:.+]], %[[ARG0_VALID:.+]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:   firrtl.connect %[[ARG0_READY:.+]], %[[ARG1_READY:.+]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:   %c42_ui64 = firrtl.constant 42 : !firrtl.uint<64>
// CHECK:   firrtl.connect %[[ARG1_DATA:.+]], %c42_ui64 : !firrtl.uint<64>, !firrtl.uint<64>
// CHECK: }

// Submodule for the ui32 ConstantOp.
// CHECK-LABEL: firrtl.module @handshake_constant_1ins_1outs_c42_ui32(
// CHECK-SAME:  in %arg0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %arg1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>) {
// CHECK:   %c42_ui32 = firrtl.constant 42 : !firrtl.uint<32>

// Submodule for the si32 ConstantOp.
// CHECK-LABEL: firrtl.module @"handshake_constant_1ins_1outs_c-11_si32"(
// CHECK-SAME:  in %arg0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %arg1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: sint<32>>) {
// CHECK:   %c-11_si32 = firrtl.constant -11 : !firrtl.sint<32>

// CHECK-LABEL: firrtl.module @test_constant(
// CHECK-SAME:  in %arg0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %arg1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %arg2: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %arg3: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out %arg4: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: sint<32>>, out %arg5: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
handshake.func @test_constant(%arg0: none, ...) -> (index, i64, ui32, si32, none) {
  %0:5 = "handshake.fork"(%arg0) {control = true} : (none) -> (none, none, none, none, none)

  // CHECK: %inst_arg0_0, %inst_arg1_1 = firrtl.instance @handshake_constant_1ins_1outs_c42_ui64  {name = ""} : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
  %1 = "handshake.constant"(%0#0) {value = 42 : index}: (none) -> index

  // CHECK: %inst_arg0_2, %inst_arg1_3 = firrtl.instance @handshake_constant_1ins_1outs_c42_ui64  {name = ""} : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
  %2 = "handshake.constant"(%0#1) {value = 42 : i64}: (none) -> i64

  // CHECK: %inst_arg0_4, %inst_arg1_5 = firrtl.instance @handshake_constant_1ins_1outs_c42_ui32  {name = ""} : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
  %3 = "handshake.constant"(%0#2) {value = 42 : ui32}: (none) -> ui32

  // CHECK: %inst_arg0_6, %inst_arg1_7 = firrtl.instance @"handshake_constant_1ins_1outs_c-11_si32"  {name = ""} : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: sint<32>>
  %4 = "handshake.constant"(%0#3) {value = -11 : si32}: (none) -> si32
  handshake.return %1, %2, %3, %4, %0#4 : index, i64, ui32, si32, none
}
