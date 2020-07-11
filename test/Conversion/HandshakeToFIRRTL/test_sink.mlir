// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s

// CHECK-LABEL: firrtl.module @handshake.sink_1ins_0outs(
// CHECK-SAME:  %arg0: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) {
// CHECK:   %0 = firrtl.subfield %arg0("ready") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.flip<uint<1>>
// CHECK:   %c1_ui1 = firrtl.constant(1 : ui1) : !firrtl.uint<1>
// CHECK:   firrtl.connect %0, %c1_ui1 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK: }

// CHECK-LABEL: firrtl.module @test_sink(
// CHECK-SAME:  %arg0: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, %arg1: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>, %arg2: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>) {
handshake.func @test_sink(%arg0: index, %arg2: none, ...) -> (none) {

  // CHECK: %0 = firrtl.instance @handshake.sink_1ins_0outs {name = ""} : !firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>>
  // CHECK: %1 = firrtl.subfield %0("arg0") : (!firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>>) -> !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>
  // CHECK: firrtl.connect %1, %arg0 : !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>
  "handshake.sink"(%arg0) : (index) -> ()

  // CHECK: firrtl.connect %arg2, %arg1 : !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>, !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>
  handshake.return %arg2 : none
}