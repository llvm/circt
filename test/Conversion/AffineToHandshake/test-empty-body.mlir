// RUN: circt-opt %s -affine-to-handshake | FileCheck %s

// This test verifies that -affine-to-handshake can create a valid 
// handshake design body from an empty function.

func @empty_body () -> () {
  return
}

// CHECK:     handshake.func @empty_body(%[[ARG_0:.*]]: none, ...) -> none {
// CHECK-NEXT:  %[[VAL_0:.*]] = "handshake.merge"(%[[ARG_0]]) : (none) -> none
// CHECK-NEXT:  "handshake.sink"(%[[VAL_0]]) : (none) -> ()
// CHECK-NEXT:  handshake.return %[[ARG_0]] : none
// CHECK-NEXT:}
