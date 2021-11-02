// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s

// CHECK-LABEL: firrtl.module @handshake_lazy_fork_in_ui64_out_ui64_ui64(
// CHECK-SAME:  in %arg0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %arg1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %arg2: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) {
// CHECK:   %0 = firrtl.subfield %arg0(0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %1 = firrtl.subfield %arg0(1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %2 = firrtl.subfield %arg0(2) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   %3 = firrtl.subfield %arg1(0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %4 = firrtl.subfield %arg1(1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %5 = firrtl.subfield %arg1(2) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   %6 = firrtl.subfield %arg2(0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %7 = firrtl.subfield %arg2(1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %8 = firrtl.subfield %arg2(2) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   %9 = firrtl.and %7, %4 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %1, %9 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:   %10 = firrtl.and %0, %9 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %3, %10 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:   firrtl.connect %5, %2 : !firrtl.uint<64>, !firrtl.uint<64>
// CHECK:   firrtl.connect %6, %10 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:   firrtl.connect %8, %2 : !firrtl.uint<64>, !firrtl.uint<64>
// CHECK: }

// CHECK-LABEL: firrtl.module @test_lazy_fork(
// CHECK-SAME:  in %arg0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>,
// CHECK-SAME:  in %arg1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>,
// CHECK-SAME:  out %arg2: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>,
// CHECK-SAME:  out %arg3: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>,
// CHECK-SAME:  out %arg4: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>,
// CHECK-SAME: in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
handshake.func @test_lazy_fork(%arg0: index, %arg1: none, ...) -> (index, index, none) {

  // CHECK: %handshake_lazy_fork_arg0, %handshake_lazy_fork_arg1, %handshake_lazy_fork_arg2 = firrtl.instance handshake_lazy_fork @handshake_lazy_fork_in_ui64_out_ui64_ui64(in arg0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out arg1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out arg2: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>)
  %0:2 = "handshake.lazy_fork"(%arg0) {control = false} : (index) -> (index, index)
  handshake.return %0#0, %0#1, %arg1 : index, index, none
}
