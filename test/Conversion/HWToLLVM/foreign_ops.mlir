// RUN: circt-opt %s --convert-hw-to-llvm | FileCheck %s

// Check that folding of non-HW operations is not attempted
// CHECK-LABEL: func.func @issue9371
func.func @issue9371(%arg: i32) -> (i32) {
  // CHECK: arith.xori
  %xor = arith.xori %arg, %arg : i32
  return %xor : i32
}
