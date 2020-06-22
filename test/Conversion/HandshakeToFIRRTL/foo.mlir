////Simple Control Path: standard
//func @foo(%arg0: i32) -> (i32) {
//  return %arg0 : i32
//}

//Simple Control Path: handshake
module {
  handshake.func @foo(%arg0: i32, %arg1: none, ...) -> (i32, none) {
    %0 = "handshake.merge"(%arg0) : (i32) -> i32
    handshake.return %0, %arg1 : i32, none
  }
}

//Simple Control Path: firrtl
//firrtl.circuit "Circuit" {
//  firrtl.module @foo(
//      %in0: !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>,
//      %in1: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>,
//      %out0: !firrtl.bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>,
//      %out1: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>) {
//    
//    %0 = firrtl.wire {name = "0"}:
//        !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>
//    firrtl.connect %0, %in0:
//        !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>,
//        !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>
//
//    firrtl.connect %out0, %0:
//        !firrtl.bundle<data: flip<sint<32>>, valid: flip<uint<1>>, ready: uint<1>>,
//        !firrtl.bundle<data: sint<32>, valid: uint<1>, ready: flip<uint<1>>>
//    
//    firrtl.connect %out1, %in1:
//        !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>,
//        !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>
//  }
//}

//Complex Control Path : standard
//func @foo(%in0: i32, %in1: i32) -> (i32) {
//  %c42 = constant 42 : i32
//  %0 = addi %in0, %in1 : i32
//  %1 = addi %0, %c42 : i32
//  return %1 : i32
//}