// RUN: circt-opt -split-input-file --handshake-add-ids %s | FileCheck %s

// CHECK: handshake.func @simple_c(%arg0: i32, %arg1: i32, %arg2: none, ...) -> (i32, none) attributes {argNames = ["arg0", "arg1", "arg2"], handshake_id = 0 : index, resNames = ["out0", "outCtrl"]} {
// CHECK:   %0 = buffer [2] %arg0 {handshake_id = 0 : index, sequential = true} : i32
// CHECK:   %1 = buffer [2] %arg1 {handshake_id = 1 : index, sequential = true} : i32
// CHECK:   %2 = arith.addi %0, %1 {handshake_id = 0 : index} : i32
// CHECK:   %3 = buffer [2] %2 {handshake_id = 2 : index, sequential = true} : i32
// CHECK:   %4 = buffer [2] %arg2 {handshake_id = 3 : index, sequential = true} : none
// CHECK:   return %3, %4 {handshake_id = 0 : index} : i32, none
// CHECK: }
handshake.func @simple_c(%arg0: i32, %arg1: i32, %arg2: none) -> (i32, none) {
  %0 = buffer [2] %arg0 {sequential = true} : i32
  %1 = buffer [2] %arg1 {sequential = true} : i32
  %2 = arith.addi %0, %1 : i32
  %3 = buffer [2] %2 {sequential = true} : i32
  %4 = buffer [2] %arg2 {sequential = true} : none
  return %3, %4 : i32, none
}
