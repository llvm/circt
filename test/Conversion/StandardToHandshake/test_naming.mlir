// RUN: circt-opt -lower-std-to-handshake %s | FileCheck %s

// CHECK-LABEL: handshake.func @main(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: none, ...) -> none attributes {argNames = ["a", "b", "c"]} {
func @main(%arg0 : i32, %b : i32, %c: i32) attributes {argNames = ["a", "b", "c"]} {
  return
}
