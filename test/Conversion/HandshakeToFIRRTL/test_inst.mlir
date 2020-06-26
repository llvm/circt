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

//module {
//  func @test_inst(%arg0: i32, %arg1: i32) -> (i32) {
//    %0 = addi %arg0, %arg1 : i32
//    return %0 : i32
//  }
//}

module {
  handshake.func @test_inst(%arg0: i32, %arg1: i32, %arg2: none, ...) -> (i32, none) {
    %0 = "handshake.merge"(%arg0) : (i32) -> i32
    %1 = "handshake.merge"(%arg1) : (i32) -> i32
    %2 = addi %0, %1 : i32
    handshake.return %2, %arg2 : i32, none
  }
}

//module {
//  firrtl.module @addi(
//      %arg0: !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>, 
//      %arg1: !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>, 
//      %result0: !firrtl.bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>) {
//
//    %arg0_data = firrtl.subfield %arg0("data") :
//        (!firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>) -> 
//        !firrtl.sint<32>
//    %arg0_valid = firrtl.subfield %arg0("valid") :
//        (!firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>) -> 
//        !firrtl.uint<1>
//    %arg0_ready = firrtl.subfield %arg0("ready") :
//        (!firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>) -> 
//        !firrtl.flip<uint<1>>
//
//    %arg1_data = firrtl.subfield %arg1("data") :
//        (!firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>) -> 
//        !firrtl.sint<32>
//    %arg1_valid = firrtl.subfield %arg1("valid") :
//        (!firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>) -> 
//        !firrtl.uint<1>
//    %arg1_ready = firrtl.subfield %arg1("ready") :
//        (!firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>) -> 
//        !firrtl.flip<uint<1>>
//
//    %result_data = firrtl.subfield %result0("data") :
//        (!firrtl.bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>) ->
//        !firrtl.flip<sint<32>>
//    %result_valid = firrtl.subfield %result0("valid") :
//        (!firrtl.bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>) ->
//        !firrtl.flip<uint<1>>
//    %result_ready = firrtl.subfield %result0("ready") :
//        (!firrtl.bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>) ->
//        !firrtl.uint<1>
//
//    %args_valid = firrtl.and %arg0_valid, %arg1_valid : 
//        (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
//    %addi_cond = firrtl.and %args_valid, %result_ready :
//        (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
//    %addi_result = firrtl.add %arg0_data, %arg1_data : 
//        (!firrtl.sint<32>, !firrtl.sint<32>) -> !firrtl.sint<32>
//
//    %c0 = firrtl.constant(0 : si32) : !firrtl.sint<32>
//    %low = firrtl.constant(0 : ui1) : !firrtl.uint<1>
//    %high = firrtl.constant(1 : ui1) : !firrtl.uint<1>
//
//    firrtl.when %addi_cond {
//      firrtl.connect %result_data, %addi_result :
//          !firrtl.flip<sint<32>>, !firrtl.sint<32>
//      firrtl.connect %result_valid, %high :
//          !firrtl.flip<uint<1>>, !firrtl.uint<1>
//      firrtl.connect %arg0_ready, %high :
//          !firrtl.flip<uint<1>>, !firrtl.uint<1>
//      firrtl.connect %arg1_ready, %high :
//          !firrtl.flip<uint<1>>, !firrtl.uint<1>
//    } else {
//      firrtl.connect %result_data, %c0 :
//          !firrtl.flip<sint<32>>, !firrtl.sint<32>
//      firrtl.connect %result_valid, %low :
//          !firrtl.flip<uint<1>>, !firrtl.uint<1>
//      firrtl.connect %arg0_ready, %low :
//          !firrtl.flip<uint<1>>, !firrtl.uint<1>
//      firrtl.connect %arg1_ready, %low :
//          !firrtl.flip<uint<1>>, !firrtl.uint<1>
//    }
//  }
//
//  firrtl.module @test_inst(
//      %arg0: !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>, 
//      %arg1: !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>, 
//      %arg2: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>, 
//      %result0: !firrtl.bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>, 
//      %result1: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>) {
//
//    %0 = firrtl.wire {name = ""} : 
//        !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>
//    firrtl.connect %0, %arg0 : 
//        !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>, 
//        !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>
//
//    %1 = firrtl.wire {name = ""} : 
//        !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>
//    firrtl.connect %1, %arg1 : 
//        !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>, 
//        !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>
//
//    %addi_inst = firrtl.instance @addi {name = ""} : !firrtl.bundle<
//        arg0: bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>, 
//        arg1: bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>, 
//        result0: bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>>
//    
//    %addi_arg0 = firrtl.subfield %addi_inst("arg0") : (!firrtl.bundle<
//        arg0: bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>, 
//        arg1: bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>, 
//        result0: bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>>) ->
//        !firrtl.bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>
//    firrtl.connect %addi_arg0, %0 : 
//        !firrtl.bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>, 
//        !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>
//
//    %addi_arg1 = firrtl.subfield %addi_inst("arg1") : (!firrtl.bundle<
//        arg0: bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>, 
//        arg1: bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>, 
//        result0: bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>>) ->
//        !firrtl.bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>
//    firrtl.connect %addi_arg1, %1 : 
//        !firrtl.bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>, 
//        !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>
//    
//    %addi_result0 = firrtl.subfield %addi_inst("result0") : (!firrtl.bundle<
//        arg0: bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>, 
//        arg1: bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>, 
//        result0: bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>>) ->
//        !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>
//
//    %2 = firrtl.wire {name = ""} : 
//        !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>
//    firrtl.connect %2, %addi_result0 : 
//        !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>, 
//        !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>
//
//    firrtl.connect %result0, %2 : 
//        !firrtl.bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>, 
//        !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>
//    firrtl.connect %result1, %arg2 : 
//        !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>, 
//        !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>
//  }
//}