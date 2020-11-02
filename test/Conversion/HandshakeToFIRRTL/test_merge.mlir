// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s

// CHECK-LABEL: firrtl.module @handshake_merge_2ins_1outs(
// CHECK-SAME:  %arg0: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, %arg1: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, %arg2: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) {
// CHECK:   %0 = firrtl.subfield %arg0("valid") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %1 = firrtl.subfield %arg0("ready") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.flip<uint<1>>
// CHECK:   %2 = firrtl.subfield %arg0("data") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   %3 = firrtl.subfield %arg1("valid") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %4 = firrtl.subfield %arg1("ready") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.flip<uint<1>>
// CHECK:   %5 = firrtl.subfield %arg1("data") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   %6 = firrtl.subfield %arg2("valid") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) -> !firrtl.flip<uint<1>>
// CHECK:   %7 = firrtl.subfield %arg2("ready") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) -> !firrtl.uint<1>
// CHECK:   %8 = firrtl.subfield %arg2("data") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) -> !firrtl.flip<uint<64>>
// CHECK:   firrtl.when %0 {
// CHECK:     firrtl.connect %8, %2 : !firrtl.flip<uint<64>>, !firrtl.uint<64>
// CHECK:     firrtl.connect %6, %0 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:     firrtl.connect %1, %7 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:   } else {
// CHECK:     firrtl.when %3 {
// CHECK:       firrtl.connect %8, %5 : !firrtl.flip<uint<64>>, !firrtl.uint<64>
// CHECK:       firrtl.connect %6, %3 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:       firrtl.connect %4, %7 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:     }
// CHECK:   }
// CHECK: }

// CHECK-LABEL: firrtl.module @test_merge(
// CHECK-SAME:  %arg0: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, %arg1: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, %arg2: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>, %arg3: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, %arg4: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>, %clock: !firrtl.clock, %reset: !firrtl.uint<1>) {
handshake.func @test_merge(%arg0: index, %arg1: index, %arg2: none, ...) -> (index, none) {

  // CHECK: %0 = firrtl.instance @handshake_merge_2ins_1outs {name = ""} : !firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, arg1: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, arg2: bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>>
  %0 = "handshake.merge"(%arg0, %arg1) : (index, index) -> index
  handshake.return %0, %arg2 : index, none
}
