// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s

// CHECK: firrtl.module @handshake.lazy_fork_1ins_2outs_ctrl(%arg0: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>, %arg1: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>, %arg2: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>) {
// CHECK:   %0 = firrtl.subfield %arg0("valid") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>) -> !firrtl.uint<1>
// CHECK:   %1 = firrtl.subfield %arg0("ready") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>) -> !firrtl.flip<uint<1>>
// CHECK:   %2 = firrtl.subfield %arg1("valid") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>) -> !firrtl.flip<uint<1>>
// CHECK:   %3 = firrtl.subfield %arg1("ready") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>) -> !firrtl.uint<1>
// CHECK:   %4 = firrtl.subfield %arg2("valid") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>) -> !firrtl.flip<uint<1>>
// CHECK:   %5 = firrtl.subfield %arg2("ready") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>) -> !firrtl.uint<1>
// CHECK:   %6 = firrtl.and %5, %3 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %1, %6 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:   %7 = firrtl.and %0, %6 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %2, %7 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:   firrtl.connect %4, %7 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK: }

// CHECK: firrtl.module @handshake.constant_1ins_1outs(%arg0: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>, %arg1: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<64>>>) {
// CHECK:   %0 = firrtl.subfield %arg0("valid") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>) -> !firrtl.uint<1>
// CHECK:   %1 = firrtl.subfield %arg0("ready") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>) -> !firrtl.flip<uint<1>>
// CHECK:   %2 = firrtl.subfield %arg1("valid") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<64>>>) -> !firrtl.flip<uint<1>>
// CHECK:   %3 = firrtl.subfield %arg1("ready") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<64>>>) -> !firrtl.uint<1>
// CHECK:   %4 = firrtl.subfield %arg1("data") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<64>>>) -> !firrtl.flip<sint<64>>
// CHECK:   firrtl.connect %2, %0 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:   firrtl.connect %1, %3 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:   %c42_si64 = firrtl.constant(42 : i64) : !firrtl.sint<64>
// CHECK:   firrtl.connect %4, %c42_si64 : !firrtl.flip<sint<64>>, !firrtl.sint<64>
// CHECK: }

// CHECK: firrtl.module @test_constant(%arg0: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>, %arg1: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<64>>>, %arg2: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>) {
handshake.func @test_constant(%arg0: none, ...) -> (index, none) {

  // CHECK: %0 = firrtl.instance @handshake.lazy_fork_1ins_2outs_ctrl {name = ""} : !firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>>, arg1: bundle<valid: uint<1>, ready: flip<uint<1>>>, arg2: bundle<valid: uint<1>, ready: flip<uint<1>>>>
  %0:2 = "handshake.lazy_fork"(%arg0) {control = true} : (none) -> (none, none)
  
  // CHECK: %4 = firrtl.instance @handshake.constant_1ins_1outs {name = ""} : !firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>>, arg1: bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<64>>>
  %1 = "handshake.constant"(%0#0) {value = 42 : index}: (none) -> index
  handshake.return %1, %0#1 : index, none
}