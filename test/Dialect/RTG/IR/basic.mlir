// RUN: circt-opt %s | FileCheck %s

// CHECK: rtg.snippet attributes {rtg.some_attr} {
rtg.snippet attributes {rtg.some_attr} {
^bb0:
}

// CHECK-LABEL: @types
// CHECK-SAME: !rtg.instruction
// CHECK-SAME: !rtg.snippet
// CHECK-SAME: !rtg.resource
func.func @types(%arg0: !rtg.instruction, %arg1: !rtg.snippet, %arg2: !rtg.resource) {
  return
}
