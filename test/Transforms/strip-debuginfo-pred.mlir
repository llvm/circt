// RUN: circt-opt -allow-unregistered-dialect %s -mlir-print-debuginfo -mlir-print-local-scope -pass-pipeline='builtin.module(strip-debuginfo-with-pred{drop-suffix=txt})' | FileCheck %s
// This test verifies that debug locations are stripped.

// CHECK-LABEL: func @inline_notation
func.func @inline_notation() {
  // CHECK: "foo"() : () -> i32 loc(unknown)
  %1 = "foo"() : () -> i32 loc("foo.txt":0:0)

// CHECK: loc(fused["foo", unknown]) 
  affine.for %i0 = 0 to 8 {
  } loc(fused["foo", "foo.txt":10:8]) 
  return
}
