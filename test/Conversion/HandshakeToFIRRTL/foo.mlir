// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s
// CHECK:       module {
// CHECK-LABEL:   firrtl.module @foo(
// CHECK-SAME:                       %[[VAL_0:.*]]: !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>, %[[VAL_1:.*]]: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>, %[[VAL_2:.*]]: !firrtl.bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>, %[[VAL_3:.*]]: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>) {
// CHECK:           %[[VAL_4:.*]] = firrtl.wire {name = ""} : !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>
// CHECK:           firrtl.connect %[[VAL_4:.*]], %[[VAL_0:.*]] : !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>, !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>
// CHECK:           firrtl.connect %[[VAL_2:.*]], %[[VAL_4:.*]] : !firrtl.bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>, !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>
// CHECK:           firrtl.connect %[[VAL_3:.*]], %[[VAL_1:.*]] : !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>, !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>
// CHECK:         }
// CHECK:       }

module {
  handshake.func @foo(%arg0: si32, %arg1: none, ...) -> (si32, none) {
    %0 = "handshake.merge"(%arg0) : (si32) -> si32
    handshake.return %0, %arg1 : si32, none
  }
}