// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s

// CHECK:       module {
// CHECK:         firrtl.circuit "simple_addi" {
// CHECK-LABEL:     firrtl.module @handshake.merge_1ins_1outs(%arg0: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>, %arg1: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>) {
// CHECK-LABEL:     firrtl.module @std.addi_2ins_1outs(%arg0: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>, %arg1: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>, %arg2: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>) {
// CHECK-LABEL:     firrtl.module @simple_addi(%arg0: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>, %arg1: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>, %arg2: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>, %arg3: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, %arg4: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>) {
// CHECK:             %[[VAL_0:.*]] = firrtl.instance @handshake.merge_1ins_1outs {name = ""} : !firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, arg1: bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>>
// CHECK:             %[[VAL_1:.*]] = firrtl.instance @handshake.merge_1ins_1outs {name = ""} : !firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, arg1: bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>>
// CHECK:             %[[VAL_2:.*]] = firrtl.instance @std.addi_2ins_1outs {name = ""} : !firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, arg1: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, arg2: bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>>
// CHECK:             firrtl.connect %arg3, %[[VAL_3:.*]] : !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>
// CHECK:             firrtl.connect %arg4, %arg2 : !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>, !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>

handshake.func @simple_addi(%arg0: i32, %arg1: i32, %arg2: none, ...) -> (i32, none) {
  %0 = "handshake.merge"(%arg0) : (i32) -> i32
  %1 = "handshake.merge"(%arg1) : (i32) -> i32
  %2 = addi %0, %1 : i32
  handshake.return %2, %arg2 : i32, none
}