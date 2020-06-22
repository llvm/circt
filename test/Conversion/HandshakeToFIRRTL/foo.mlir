////Simple Control Path: standard
//func @foo(%arg0: si32) -> (si32) {
//  return %arg0 : si32
//}

//Simple Control Path: handshake
module {
  handshake.func @foo(%arg0: si32, %arg1: none, ...) -> (si32, none) {
    %0 = "handshake.merge"(%arg0) : (si32) -> si32
    handshake.return %0, %arg1 : si32, none
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
//func @foo(%in0: si32, %in1: si32) -> (si32) {
//  %c42 = constant 42 : si32
//  %0 = addi %in0, %in1 : si32
//  %1 = addi %0, %c42 : si32
//  return %1 : si32
//}