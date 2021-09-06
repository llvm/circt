// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s

// CHECK-LABEL: firrtl.module @handshake_join_2ins_1outs_ctrl(
// CHECK-SAME:  in %arg0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>,
// CHECK-SAME:  in %arg1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>,
// CHECK-SAME:  out %arg2: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>) {
 // CHECK:   %0 = firrtl.subfield %arg0(0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>>) -> !firrtl.uint<1>
 // CHECK:   %1 = firrtl.subfield %arg0(1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>>) -> !firrtl.uint<1>
 // CHECK:   %2 = firrtl.subfield %arg1(0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>>) -> !firrtl.uint<1>
 // CHECK:   %3 = firrtl.subfield %arg1(1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>>) -> !firrtl.uint<1>
 // CHECK:   %4 = firrtl.subfield %arg2(0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>>) -> !firrtl.uint<1>
 // CHECK:   %5 = firrtl.subfield %arg2(1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>>) -> !firrtl.uint<1>
 // CHECK:   %6 = firrtl.and %2, %0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
 // CHECK:   firrtl.connect %4, %6 : !firrtl.uint<1>, !firrtl.uint<1>
 // CHECK:   %7 = firrtl.and %5, %6 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
 // CHECK:   firrtl.connect %1, %7 : !firrtl.uint<1>, !firrtl.uint<1>
 // CHECK:   firrtl.connect %3, %7 : !firrtl.uint<1>, !firrtl.uint<1>
 // CHECK: }

// CHECK-LABEL: firrtl.module @test_join(
// CHECK-SAME: in %arg0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %arg1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %arg2: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %arg3: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %arg4: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
handshake.func @test_join(%arg0: none, %arg1: none, %arg2: none, ...) -> (none, none) {

  // CHECK: %inst_arg0, %inst_arg1, %inst_arg2 = firrtl.instance @handshake_join_2ins_1outs_ctrl {name = ""} : in !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
  %0 = "handshake.join"(%arg0, %arg1) {control = true}: (none, none) -> none
  handshake.return %0, %arg2 : none, none
}
