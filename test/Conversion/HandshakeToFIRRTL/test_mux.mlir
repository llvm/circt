// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s

// CHECK-LABEL: firrtl.module @handshake.mux_3ins_1outs(
// CHECK-SAME:  %arg0: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, %arg1: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, %arg2: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, %arg3: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) {
// CHECK:   %0 = firrtl.subfield %arg0("valid") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %1 = firrtl.subfield %arg0("ready") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.flip<uint<1>>
// CHECK:   %2 = firrtl.subfield %arg0("data") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   %3 = firrtl.subfield %arg1("valid") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %4 = firrtl.subfield %arg1("ready") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.flip<uint<1>>
// CHECK:   %5 = firrtl.subfield %arg1("data") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   %6 = firrtl.subfield %arg2("valid") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %7 = firrtl.subfield %arg2("ready") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.flip<uint<1>>
// CHECK:   %8 = firrtl.subfield %arg2("data") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   %9 = firrtl.subfield %arg3("valid") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) -> !firrtl.flip<uint<1>>
// CHECK:   %10 = firrtl.subfield %arg3("ready") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) -> !firrtl.uint<1>
// CHECK:   %11 = firrtl.subfield %arg3("data") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) -> !firrtl.flip<uint<64>>
// CHECK:   firrtl.when %0 {
// CHECK:     %c1_ui64 = firrtl.constant(1 : ui64) : !firrtl.uint<64>
// CHECK:     %12 = firrtl.eq %2, %c1_ui64 : (!firrtl.uint<64>, !firrtl.uint<64>) -> !firrtl.uint<1>
// CHECK:     firrtl.when %12 {
// CHECK:       firrtl.connect %9, %3 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:       firrtl.connect %11, %5 : !firrtl.flip<uint<64>>, !firrtl.uint<64>
// CHECK:       firrtl.connect %4, %10 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:       %13 = firrtl.and %3, %10 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:       firrtl.connect %1, %13 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:     } else {
// CHECK:       %c2_ui64 = firrtl.constant(2 : ui64) : !firrtl.uint<64>
// CHECK:       %13 = firrtl.eq %2, %c2_ui64 : (!firrtl.uint<64>, !firrtl.uint<64>) -> !firrtl.uint<1>
// CHECK:       firrtl.when %13 {
// CHECK:         firrtl.connect %9, %6 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:         firrtl.connect %11, %8 : !firrtl.flip<uint<64>>, !firrtl.uint<64>
// CHECK:         firrtl.connect %7, %10 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:         %14 = firrtl.and %6, %10 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:         firrtl.connect %1, %14 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:       }
// CHECK:     }
// CHECK:   }
// CHECK: }

// CHECK: firrtl.module @test_mux(%arg0: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, %arg1: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, %arg2: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, %arg3: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>, %arg4: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, %arg5: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>, %clock: !firrtl.clock, %reset: !firrtl.uint<1>) {
handshake.func @test_mux(%arg0: index, %arg1: index, %arg2: index, %arg3: none, ...) -> (index, none) {

  // CHECK: %0 = firrtl.instance @handshake.mux_3ins_1outs {name = ""} : !firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, arg1: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, arg2: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, arg3: bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>>
  %0 = "handshake.mux"(%arg0, %arg1, %arg2): (index, index, index) -> index
  handshake.return %0, %arg3 : index, none
}