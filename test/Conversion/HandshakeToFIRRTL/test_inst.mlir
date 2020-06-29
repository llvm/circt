// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s
// CHECK:       module {
// CHECK-LABEL:   firrtl.module @std.addi(
// CHECK-SAME:                            %[[VAL_0:.*]]: !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>, %[[VAL_1:.*]]: !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>, %[[VAL_2:.*]]: !firrtl.bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>) {
// CHECK:           %[[VAL_3:.*]] = firrtl.subfield %[[VAL_0:.*]]("data") : (!firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>) -> !firrtl.sint<32>
// CHECK:           %[[VAL_4:.*]] = firrtl.subfield %[[VAL_0:.*]]("valid") : (!firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>) -> !firrtl.uint<1>
// CHECK:           %[[VAL_5:.*]] = firrtl.subfield %[[VAL_0:.*]]("ready") : (!firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>) -> !firrtl.flip<uint<1>>
// CHECK:           %[[VAL_6:.*]] = firrtl.subfield %[[VAL_1:.*]]("data") : (!firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>) -> !firrtl.sint<32>
// CHECK:           %[[VAL_7:.*]] = firrtl.subfield %[[VAL_1:.*]]("valid") : (!firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>) -> !firrtl.uint<1>
// CHECK:           %[[VAL_8:.*]] = firrtl.subfield %[[VAL_1:.*]]("ready") : (!firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>) -> !firrtl.flip<uint<1>>
// CHECK:           %[[VAL_9:.*]] = firrtl.subfield %[[VAL_2:.*]]("data") : (!firrtl.bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>) -> !firrtl.flip<sint<32>>
// CHECK:           %[[VAL_10:.*]] = firrtl.subfield %[[VAL_2:.*]]("valid") : (!firrtl.bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>) -> !firrtl.flip<uint<1>>
// CHECK:           %[[VAL_11:.*]] = firrtl.subfield %[[VAL_2:.*]]("ready") : (!firrtl.bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>) -> !firrtl.uint<1>
// CHECK:           %[[VAL_12:.*]] = firrtl.constant(0 : si32) : !firrtl.sint<32>
// CHECK:           %[[VAL_13:.*]] = firrtl.constant(0 : ui1) : !firrtl.uint<1>
// CHECK:           %[[VAL_14:.*]] = firrtl.constant(1 : ui1) : !firrtl.uint<1>
// CHECK:           %[[VAL_15:.*]] = firrtl.and %[[VAL_4:.*]], %[[VAL_7:.*]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:           %[[VAL_16:.*]] = firrtl.and %[[VAL_15:.*]], %[[VAL_11:.*]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:           %[[VAL_17:.*]] = firrtl.add %[[VAL_3:.*]], %[[VAL_6:.*]] : (!firrtl.sint<32>, !firrtl.sint<32>) -> !firrtl.sint<32>
// CHECK:           firrtl.when %[[VAL_16:.*]] {
// CHECK:             firrtl.connect %[[VAL_9:.*]], %[[VAL_17:.*]] : !firrtl.flip<sint<32>>, !firrtl.sint<32>
// CHECK:             firrtl.connect %[[VAL_10:.*]], %[[VAL_14:.*]] : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_5:.*]], %[[VAL_14:.*]] : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_8:.*]], %[[VAL_14:.*]] : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:           } else {
// CHECK:             firrtl.connect %[[VAL_9:.*]], %[[VAL_12:.*]] : !firrtl.flip<sint<32>>, !firrtl.sint<32>
// CHECK:             firrtl.connect %[[VAL_10:.*]], %[[VAL_13:.*]] : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_5:.*]], %[[VAL_13:.*]] : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_8:.*]], %[[VAL_13:.*]] : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:           }
// CHECK:         }
// CHECK-LABEL:   firrtl.module @test_inst(
// CHECK-SAME:                             %[[VAL_18:.*]]: !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>, %[[VAL_19:.*]]: !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>, %[[VAL_20:.*]]: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>, %[[VAL_21:.*]]: !firrtl.bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>, %[[VAL_22:.*]]: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>) {
// CHECK:           %[[VAL_23:.*]] = firrtl.wire {name = ""} : !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>
// CHECK:           firrtl.connect %[[VAL_23:.*]], %[[VAL_18:.*]] : !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>, !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>
// CHECK:           %[[VAL_24:.*]] = firrtl.wire {name = ""} : !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>
// CHECK:           firrtl.connect %[[VAL_24:.*]], %[[VAL_19:.*]] : !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>, !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>
// CHECK:           %[[VAL_25:.*]] = firrtl.instance @std.addi {name = ""} : !firrtl.bundle<arg0: bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>, arg1: bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>, arg2: bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>>
// CHECK:           %[[VAL_26:.*]] = firrtl.subfield %[[VAL_25:.*]]("arg0") : (!firrtl.bundle<arg0: bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>, arg1: bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>, arg2: bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>>) -> !firrtl.bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>
// CHECK:           firrtl.connect %[[VAL_26:.*]], %[[VAL_23:.*]] : !firrtl.bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>, !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>
// CHECK:           %[[VAL_27:.*]] = firrtl.subfield %[[VAL_25:.*]]("arg1") : (!firrtl.bundle<arg0: bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>, arg1: bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>, arg2: bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>>) -> !firrtl.bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>
// CHECK:           firrtl.connect %[[VAL_27:.*]], %[[VAL_24:.*]] : !firrtl.bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>, !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>
// CHECK:           %[[VAL_28:.*]] = firrtl.subfield %[[VAL_25:.*]]("arg2") : (!firrtl.bundle<arg0: bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>, arg1: bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>, arg2: bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>>) -> !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>
// CHECK:           %[[VAL_29:.*]] = firrtl.wire {name = ""} : !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>
// CHECK:           firrtl.connect %[[VAL_29:.*]], %[[VAL_28:.*]] : !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>, !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>
// CHECK:           firrtl.connect %[[VAL_21:.*]], %[[VAL_29:.*]] : !firrtl.bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>, !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>
// CHECK:           firrtl.connect %[[VAL_22:.*]], %[[VAL_20:.*]] : !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>, !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>
// CHECK:         }
// CHECK:       }

module {
  handshake.func @test_inst(%arg0: i32, %arg1: i32, %arg2: none, ...) -> (i32, none) {
    %0 = "handshake.merge"(%arg0) : (i32) -> i32
    %1 = "handshake.merge"(%arg1) : (i32) -> i32
    %2 = addi %0, %1 : i32
    handshake.return %2, %arg2 : i32, none
  }
}