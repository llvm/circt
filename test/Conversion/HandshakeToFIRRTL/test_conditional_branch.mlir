// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s

// CHECK-LABEL: firrtl.module @handshake_conditional_branch_in_ui1_ui64_out_ui64_ui64(
// CHECK-SAME:  in %[[ARG0:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<1>>, in %[[ARG1:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %[[ARG2:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %[[ARG3:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) {
// CHECK:   %0 = firrtl.subfield %[[ARG0]](0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<1>>) -> !firrtl.uint<1>
// CHECK:   %1 = firrtl.subfield %[[ARG0]](1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<1>>) -> !firrtl.uint<1>
// CHECK:   %2 = firrtl.subfield %[[ARG0]](2) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<1>>) -> !firrtl.uint<1>
// CHECK:   %3 = firrtl.subfield %[[ARG1]](0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %4 = firrtl.subfield %[[ARG1]](1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %5 = firrtl.subfield %[[ARG1]](2) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   %6 = firrtl.subfield %[[ARG2]](0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %7 = firrtl.subfield %[[ARG2]](1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %8 = firrtl.subfield %[[ARG2]](2) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   %9 = firrtl.subfield %[[ARG3]](0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %10 = firrtl.subfield %[[ARG3]](1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %11 = firrtl.subfield %[[ARG3]](2) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   %12 = firrtl.and %0, %3 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   %13 = firrtl.not %2 : (!firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   %14 = firrtl.and %2, %12 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %6, %14 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:   %15 = firrtl.and %13, %12 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %9, %15 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:   firrtl.connect %8, %5 : !firrtl.uint<64>, !firrtl.uint<64>
// CHECK:   firrtl.connect %11, %5 : !firrtl.uint<64>, !firrtl.uint<64>
// CHECK:   %16 = firrtl.mux(%2, %7, %10) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   %17 = firrtl.and %16, %12 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %4, %17 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:   firrtl.connect %1, %17 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK: }

// CHECK: firrtl.module @test_conditional_branch(in %[[VAL_22:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<1>>, in %[[VAL_23:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_24:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_25:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %[[VAL_26:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %[[VAL_27:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_28:.*]]: !firrtl.clock, in %[[VAL_29:.*]]: !firrtl.uint<1>) {
// CHECK:   %[[VAL_30:.*]], %[[VAL_31:.*]], %[[VAL_32:.*]], %[[VAL_33:.*]] = firrtl.instance handshake_conditional_branch0  @handshake_conditional_branch_in_ui1_ui64_out_ui64_ui64(in [[ARG0:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<1>>, in [[ARG1:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out [[ARG2:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out [[ARG3:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>)
handshake.func @test_conditional_branch(%arg0: i1, %arg1: index, %arg2: none, ...) -> (index, index, none) {
  %0:2 = "handshake.conditional_branch"(%arg0, %arg1) {control = false}: (i1, index) -> (index, index)
  handshake.return %0#0, %0#1, %arg2 : index, index, none
}
