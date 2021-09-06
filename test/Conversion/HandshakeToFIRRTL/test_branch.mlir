// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s

// CHECK-LABEL: firrtl.module @handshake_branch_1ins_1outs_ui64(
// CHECK-SAME: in %arg0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>,
// CHECK-SAME: out %arg1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) {
// CHECK:   %0 = firrtl.subfield %arg0(0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %1 = firrtl.subfield %arg0(1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %2 = firrtl.subfield %arg0(2) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   %3 = firrtl.subfield %arg1(0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %4 = firrtl.subfield %arg1(1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %5 = firrtl.subfield %arg1(2) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   firrtl.connect %3, %0 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:   firrtl.connect %1, %4 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:   firrtl.connect %5, %2 : !firrtl.uint<64>, !firrtl.uint<64>
// CHECK: }

// CHECK-LABEL: firrtl.module @test_branch(
// CHECK-SAME:  in %arg0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %arg1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %arg2: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %arg3: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
handshake.func @test_branch(%arg0: index, %arg1: none, ...) -> (index, none) {

  // CHECK: %inst_arg0, %inst_arg1 = firrtl.instance @handshake_branch_1ins_1outs_ui64 {name = ""} : in !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
  %0 = "handshake.branch"(%arg0) {control = false}: (index) -> index
  handshake.return %0, %arg1 : index, none
}
