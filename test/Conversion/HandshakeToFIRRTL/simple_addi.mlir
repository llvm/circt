// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s

// CHECK:       module {
// CHECK:         firrtl.circuit "simple_addi" {
// CHECK-LABEL:     firrtl.module @handshake.merge_1ins_1outs(%arg0: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>, %arg1: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>) {
// CHECK:             %[[VAL_0:.*]] = firrtl.subfield %arg0("valid") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_1:.*]] = firrtl.subfield %arg0("ready") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>) -> !firrtl.flip<uint<1>>
// CHECK:             %[[VAL_2:.*]] = firrtl.subfield %arg0("data") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>) -> !firrtl.sint<32>
// CHECK:             %[[VAL_3:.*]] = firrtl.subfield %arg1("valid") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>) -> !firrtl.flip<uint<1>>
// CHECK:             %[[VAL_4:.*]] = firrtl.subfield %arg1("ready") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_5:.*]] = firrtl.subfield %arg1("data") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>) -> !firrtl.flip<sint<32>>
// CHECK:             firrtl.connect %[[VAL_1:.*]], %[[VAL_4:.*]] : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_5:.*]], %[[VAL_2:.*]] : !firrtl.flip<sint<32>>, !firrtl.sint<32>
// CHECK:             firrtl.connect %[[VAL_3:.*]], %[[VAL_0:.*]] : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:           }
// CHECK-LABEL:     firrtl.module @std.addi_2ins_1outs(%arg0: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>, %arg1: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>, %arg2: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>) {
// CHECK:             %[[VAL_0:.*]] = firrtl.subfield %arg0("valid") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_1:.*]] = firrtl.subfield %arg0("ready") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>) -> !firrtl.flip<uint<1>>
// CHECK:             %[[VAL_2:.*]] = firrtl.subfield %arg0("data") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>) -> !firrtl.sint<32>
// CHECK:             %[[VAL_3:.*]] = firrtl.subfield %arg1("valid") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_4:.*]] = firrtl.subfield %arg1("ready") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>) -> !firrtl.flip<uint<1>>
// CHECK:             %[[VAL_5:.*]] = firrtl.subfield %arg1("data") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>) -> !firrtl.sint<32>
// CHECK:             %[[VAL_6:.*]] = firrtl.subfield %arg2("valid") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>) -> !firrtl.flip<uint<1>>
// CHECK:             %[[VAL_7:.*]] = firrtl.subfield %arg2("ready") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_8:.*]] = firrtl.subfield %arg2("data") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>) -> !firrtl.flip<sint<32>>
// CHECK:             %[[VAL_9:.*]] = firrtl.add %[[VAL_2:.*]], %[[VAL_5:.*]] : (!firrtl.sint<32>, !firrtl.sint<32>) -> !firrtl.sint<32>
// CHECK:             firrtl.connect %[[VAL_8:.*]], %[[VAL_9:.*]] : !firrtl.flip<sint<32>>, !firrtl.sint<32>
// CHECK:             %[[VAL_10:.*]] = firrtl.and %[[VAL_0:.*]], %[[VAL_3:.*]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_6:.*]], %[[VAL_10:.*]] : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:             %[[VAL_11:.*]] = firrtl.and %[[VAL_7:.*]], %[[VAL_10:.*]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_1:.*]], %[[VAL_11:.*]] : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_4:.*]], %[[VAL_11:.*]] : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:           }
// CHECK-LABEL:     firrtl.module @simple_addi(%arg0: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>, %arg1: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>, %arg2: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>, %arg3: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>, %arg4: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, %arg5: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>) {
// CHECK:             %[[VAL_0:.*]] = firrtl.instance @handshake.merge_1ins_1outs {name = ""} : !firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, arg1: bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>>
// CHECK:             %[[VAL_1:.*]] = firrtl.subfield %[[VAL_0:.*]]("arg0") : (!firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, arg1: bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>>) -> !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>
// CHECK:             firrtl.connect %[[VAL_1:.*]], %arg0 : !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>
// CHECK:             %[[VAL_2:.*]] = firrtl.subfield %[[VAL_0:.*]]("arg1") : (!firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, arg1: bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>>) -> !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>
// CHECK:             %[[VAL_3:.*]] = firrtl.instance @handshake.merge_1ins_1outs {name = ""} : !firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, arg1: bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>>
// CHECK:             %[[VAL_4:.*]] = firrtl.subfield %[[VAL_3:.*]]("arg0") : (!firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, arg1: bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>>) -> !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>
// CHECK:             firrtl.connect %[[VAL_4:.*]], %arg1 : !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>
// CHECK:             %[[VAL_5:.*]] = firrtl.subfield %[[VAL_3:.*]]("arg1") : (!firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, arg1: bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>>) -> !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>
// CHECK:             %[[VAL_6:.*]] = firrtl.instance @handshake.merge_1ins_1outs {name = ""} : !firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, arg1: bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>>
// CHECK:             %[[VAL_7:.*]] = firrtl.subfield %[[VAL_6:.*]]("arg0") : (!firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, arg1: bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>>) -> !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>
// CHECK:             firrtl.connect %[[VAL_7:.*]], %arg2 : !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>
// CHECK:             %[[VAL_8:.*]] = firrtl.subfield %[[VAL_6:.*]]("arg1") : (!firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, arg1: bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>>) -> !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>
// CHECK:             %[[VAL_9:.*]] = firrtl.instance @std.addi_2ins_1outs {name = ""} : !firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, arg1: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, arg2: bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>>
// CHECK:             %[[VAL_10:.*]] = firrtl.subfield %[[VAL_9:.*]]("arg0") : (!firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, arg1: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, arg2: bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>>) -> !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>
// CHECK:             firrtl.connect %[[VAL_10:.*]], %[[VAL_2:.*]] : !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>
// CHECK:             %[[VAL_11:.*]] = firrtl.subfield %[[VAL_9:.*]]("arg1") : (!firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, arg1: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, arg2: bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>>) -> !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>
// CHECK:             firrtl.connect %[[VAL_11:.*]], %[[VAL_5:.*]] : !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>
// CHECK:             %[[VAL_12:.*]] = firrtl.subfield %[[VAL_9:.*]]("arg2") : (!firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, arg1: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, arg2: bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>>) -> !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>
// CHECK:             %[[VAL_13:.*]] = firrtl.instance @std.addi_2ins_1outs {name = ""} : !firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, arg1: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, arg2: bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>>
// CHECK:             %[[VAL_14:.*]] = firrtl.subfield %[[VAL_13:.*]]("arg0") : (!firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, arg1: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, arg2: bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>>) -> !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>
// CHECK:             firrtl.connect %[[VAL_14:.*]], %[[VAL_8:.*]] : !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>
// CHECK:             %[[VAL_15:.*]] = firrtl.subfield %[[VAL_13:.*]]("arg1") : (!firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, arg1: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, arg2: bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>>) -> !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>
// CHECK:             firrtl.connect %[[VAL_15:.*]], %[[VAL_12:.*]] : !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>
// CHECK:             %[[VAL_16:.*]] = firrtl.subfield %[[VAL_13:.*]]("arg2") : (!firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, arg1: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, arg2: bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>>) -> !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>
// CHECK:             firrtl.connect %arg4, %[[VAL_16:.*]] : !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<sint<32>>>, !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: sint<32>>
// CHECK:             firrtl.connect %arg5, %arg3 : !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>, !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>
// CHECK:           }
// CHECK:         }
// CHECK:       }

module {
  handshake.func @simple_addi(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: none, ...) -> (i32, none) {
    %0 = "handshake.merge"(%arg0) : (i32) -> i32
    %1 = "handshake.merge"(%arg1) : (i32) -> i32
    %2 = "handshake.merge"(%arg2) : (i32) -> i32
    %3 = addi %0, %1 : i32
    %4 = addi %2, %3 : i32
    handshake.return %4, %arg3 : i32, none
  }
}