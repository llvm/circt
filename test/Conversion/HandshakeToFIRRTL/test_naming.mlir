// RUN: circt-opt -split-input-file -lower-handshake-to-firrtl %s | FileCheck %s

// CHECK-LABEL: firrtl.module @main(
// CHECK:  in %a: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, 
// CHECK:  in %b: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, 
// CHECK:  in %inCtrl: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, 
// CHECK:  out %out0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, 
// CHECK:  out %outCtrl: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, 
// CHECK:  in %clock: !firrtl.clock, 
// CHECK:  in %reset: !firrtl.uint<1>) {
handshake.func @main(%a: index, %b: index, %inCtrl: none, ...) -> (index, none) {
  %0 = arith.addi %a, %b : index
  return %0, %inCtrl : index, none
}

// -----

// CHECK-LABEL: firrtl.module @main(
// CHECK:  in %aTest: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, 
// CHECK:  in %bTest: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, 
// CHECK:  in %cTest: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, 
// CHECK:  out %outTest: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, 
// CHECK:  out %coutTest: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, 
// CHECK:  in %clock: !firrtl.clock, 
// CHECK:  in %reset: !firrtl.uint<1>) {
handshake.func @main(%a: index, %b: index, %inCtrl: none, ...) -> (index, none) attributes {argNames = ["aTest", "bTest", "cTest"], resNames = ["outTest", "coutTest"]} {
  %0 = arith.addi %a, %b : index
  return %0, %inCtrl : index, none
}


// -----

// CHECK-LABEL: firrtl.module @handshake_mux_in_ui64_ui64_ui64_out_ui64(
// CHECK:         in %select: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, 
// CHECK:         in %in0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, 
// CHECK:         in %in1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, 
// CHECK:         out %out0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) {

// CHECK-LABEL: firrtl.module @test_mux(
// CHECK: %handshake_mux0_select, %handshake_mux0_in0, %handshake_mux0_in1, %handshake_mux0_out0 = firrtl.instance handshake_mux0  @handshake_mux_in_ui64_ui64_ui64_out_ui64(in select: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in in0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in in1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out out0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>)
handshake.func @test_mux(%arg0: index, %arg1: index, %arg2: index, %arg3: none, ...) -> (index, none) {
  %0 = mux %arg0 [%arg1, %arg2] : index, index
  return %0, %arg3 : index, none
}

// -----

// External memories can be quite verbose due to the nested bundles, so make
// sure that we generate meaningful names for them.

// CHECK-LABEL: firrtl.module @main(
// CHECK:   in %ldAddr: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, 
// CHECK:   in %stAddr: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, 
// CHECK:   in %v: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, 
// CHECK:   in %mem: !firrtl.bundle<
// CHECK:     stData0 flip: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, 
// CHECK:     stAddr0 flip: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, 
// CHECK:     ldAddr0 flip: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, 
// CHECK:     lddata0: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, 
// CHECK:     stDone0: bundle<valid: uint<1>, ready flip: uint<1>>, 
// CHECK:     ldDone0: bundle<valid: uint<1>, ready flip: uint<1>>>, 
// CHECK:   in %argCtrl: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, 
// CHECK:   out %outCtrl: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, 
// CHECK:   in %clock: !firrtl.clock, 
// CHECK:   in %reset: !firrtl.uint<1>) {
// CHECK: %handshake_extmemory0_extmem, %handshake_extmemory0_stData0, %handshake_extmemory0_stAddr0, %handshake_extmemory0_ldAddr0, %handshake_extmemory0_lddata0, %handshake_extmemory0_stDone0, %handshake_extmemory0_ldDone0 = firrtl.instance handshake_extmemory0  @handshake_extmemory_in_ui32_ui64_ui64_out_ui32(in extmem: !firrtl.bundle<stData0 flip: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, stAddr0 flip: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, ldAddr0 flip: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, lddata0: bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, stDone0: bundle<valid: uint<1>, ready flip: uint<1>>, ldDone0: bundle<valid: uint<1>, ready flip: uint<1>>>, in stData0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in stAddr0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in ldAddr0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out lddata0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out stDone0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out ldDone0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>)

handshake.func @main(%ldAddr: index, %stAddr: index, %v: i32, %mem : memref<10xi32>, %argCtrl: none) -> none {
  %ldData, %stCtrl, %ldCtrl = handshake.extmemory[ld=1, st=1](%mem : memref<10xi32>)(%storeData, %storeAddr, %loadAddr) {id = 0 : i32} : (i32, index, index) -> (i32, none, none)
  %fCtrl:2 = fork [2] %argCtrl : none
  %loadData, %loadAddr = load [%ldAddr] %ldData, %fCtrl#0 : index, i32
  %storeData, %storeAddr = store [%stAddr] %v, %fCtrl#1 : index, i32
  sink %loadData : i32
  %finCtrl = join %stCtrl, %ldCtrl : none
  return %finCtrl : none
}
