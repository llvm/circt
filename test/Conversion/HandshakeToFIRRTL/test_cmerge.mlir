// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s

// CHECK-LABEL: firrtl.module @handshake_control_merge_2ins_2outs_ctrl(
// CHECK-SAME:  %arg0: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>, %arg1: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>, %arg2: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>, %arg3: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) {
// CHECK:   %0 = firrtl.subfield %arg0("valid") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>) -> !firrtl.uint<1>
// CHECK:   %1 = firrtl.subfield %arg0("ready") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>) -> !firrtl.flip<uint<1>>
// CHECK:   %2 = firrtl.subfield %arg1("valid") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>) -> !firrtl.uint<1>
// CHECK:   %3 = firrtl.subfield %arg1("ready") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>) -> !firrtl.flip<uint<1>>
// CHECK:   %4 = firrtl.subfield %arg2("valid") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>) -> !firrtl.flip<uint<1>>
// CHECK:   %5 = firrtl.subfield %arg2("ready") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>) -> !firrtl.uint<1>
// CHECK:   %6 = firrtl.subfield %arg3("valid") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) -> !firrtl.flip<uint<1>>
// CHECK:   %7 = firrtl.subfield %arg3("ready") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) -> !firrtl.uint<1>
// CHECK:   %8 = firrtl.subfield %arg3("data") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) -> !firrtl.flip<uint<64>>
// CHECK:   %9 = firrtl.and %5, %7 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.when %0 {
// CHECK:     %c0_ui64 = firrtl.constant(0 : ui64) : !firrtl.uint<64>
// CHECK:     firrtl.connect %8, %c0_ui64 : !firrtl.flip<uint<64>>, !firrtl.uint<64>
// CHECK:     firrtl.connect %6, %0 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:     firrtl.connect %4, %0 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:     firrtl.connect %1, %9 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:   } else {
// CHECK:     firrtl.when %2 {
// CHECK:       %c1_ui64 = firrtl.constant(1 : ui64) : !firrtl.uint<64>
// CHECK:       firrtl.connect %8, %c1_ui64 : !firrtl.flip<uint<64>>, !firrtl.uint<64>
// CHECK:       firrtl.connect %6, %2 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:       firrtl.connect %4, %2 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:       firrtl.connect %3, %9 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:     }
// CHECK:   }
// CHECK: }

// CHECK-LABEL: firrtl.module @test_cmerge(
// CHECK-SAME:  %arg0: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>, %arg1: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>, %arg2: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>, %arg3: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>, %arg4: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, %arg5: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>, %clock: !firrtl.clock, %reset: !firrtl.uint<1>) {
handshake.func @test_cmerge(%arg0: none, %arg1: none, %arg2: none, ...) -> (none, index, none) {

  // CHECK: %0 = firrtl.instance @handshake_control_merge_2ins_2outs_ctrl {name = ""} : !firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>>, arg1: bundle<valid: flip<uint<1>>, ready: uint<1>>, arg2: bundle<valid: uint<1>, ready: flip<uint<1>>>, arg3: bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>>
  %0:2 = "handshake.control_merge"(%arg0, %arg1) {control = true} : (none, none) -> (none, index)
  handshake.return %0#0, %0#1, %arg2 : none, index, none
}
