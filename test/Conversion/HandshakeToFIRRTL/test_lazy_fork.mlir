// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s

// CHECK: firrtl.module @handshake.lazy_fork_1ins_2outs(%arg0: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<64>>, %arg1: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<64>>>, %arg2: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<64>>>) {
// CHECK:   %0 = firrtl.subfield %arg0("valid") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<64>>) -> !firrtl.uint<1>
// CHECK:   %1 = firrtl.subfield %arg0("ready") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<64>>) -> !firrtl.flip<uint<1>>
// CHECK:   %2 = firrtl.subfield %arg0("data") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<64>>) -> !firrtl.sint<64>
// CHECK:   %3 = firrtl.subfield %arg1("valid") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<64>>>) -> !firrtl.flip<uint<1>>
// CHECK:   %4 = firrtl.subfield %arg1("ready") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<64>>>) -> !firrtl.uint<1>
// CHECK:   %5 = firrtl.subfield %arg1("data") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<64>>>) -> !firrtl.flip<sint<64>>
// CHECK:   %6 = firrtl.subfield %arg2("valid") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<64>>>) -> !firrtl.flip<uint<1>>
// CHECK:   %7 = firrtl.subfield %arg2("ready") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<64>>>) -> !firrtl.uint<1>
// CHECK:   %8 = firrtl.subfield %arg2("data") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<64>>>) -> !firrtl.flip<sint<64>>
// CHECK:   %9 = firrtl.and %7, %4 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %1, %9 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:   %10 = firrtl.and %0, %9 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %3, %10 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:   firrtl.connect %5, %2 : !firrtl.flip<sint<64>>, !firrtl.sint<64>
// CHECK:   firrtl.connect %6, %10 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:   firrtl.connect %8, %2 : !firrtl.flip<sint<64>>, !firrtl.sint<64>
// CHECK: }

// CHECK: firrtl.module @test_lazy_fork(%arg0: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<64>>, %arg1: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>, %arg2: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<64>>>, %arg3: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<64>>>, %arg4: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>) {
handshake.func @test_lazy_fork(%arg0: index, %arg1: none, ...) -> (index, index, none) {

  // CHECK: %0 = firrtl.instance @handshake.lazy_fork_1ins_2outs {name = ""} : !firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<64>>>, arg1: bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<64>>, arg2: bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<64>>>
  %0:2 = "handshake.lazy_fork"(%arg0) {control = false} : (index) -> (index, index)
  handshake.return %0#0, %0#1, %arg1 : index, index, none
}