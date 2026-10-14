// RUN: circt-opt --split-input-file --lower-handshake-to-hw %s -verify-diagnostics

// expected-error @+1 {{'handshake.func' op error during conversion}}
handshake.func @main(%arg0: index, %arg1: index, %v: i32, %mem : memref<10xi32>, %argCtrl: none) -> none {
  // expected-error @+1 {{failed to legalize operation 'handshake.extmemory' that was explicitly marked illegal}}
  %ldData, %stCtrl, %ldCtrl = handshake.extmemory[ld=1, st=1](%mem : memref<10xi32>)(%storeData, %storeAddr, %loadAddr) {id = 0 : i32} : (i32, index, index) -> (i32, none, none)
  %fCtrl:2 = fork [2] %argCtrl : none
  %loadData, %loadAddr = load [%arg0] %ldData, %fCtrl#0 : index, i32
  %storeData, %storeAddr = store [%arg1] %v, %fCtrl#1 : index, i32
  sink %loadData : i32
  %finCtrl = join %stCtrl, %ldCtrl : none, none
  return %finCtrl : none
}

// -----

// expected-error @+1 {{HandshakeToHW: failed to verify that all values are used exactly once. Remember to run the fork/sink materialization pass before HW lowering.}}
handshake.func @main() -> () {
// expected-error @+1 {{'handshake.source' op result 0 has no uses.}}
  %0 = source
  return
}

// -----

// This lowering materializes constants as hw.constant, which only exists
// for integer types, so a floating-point constant has to be rejected.
// Before it was, the failed IntegerAttr cast was dereferenced unchecked
// and the pass segfaulted.

// expected-error @+1 {{'handshake.func' op error during conversion}}
handshake.func @float_constant(%arg0: none, ...) -> (f32, none) {
  %0:2 = fork [2] %arg0 : none
  // expected-error @+2 {{lowering to HW only supports integer constants, got '0.000000e+00 : f32'}}
  // expected-error @+1 {{failed to legalize operation 'handshake.constant' that was explicitly marked illegal}}
  %1 = constant %0#0 {value = 0.000000e+00 : f32} : f32
  return %1, %0#1 : f32, none
}
