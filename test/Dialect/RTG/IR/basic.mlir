// RUN: circt-opt %s | FileCheck %s

// CHECK: [[SNIPPET:%.+]] = rtg.snippet attributes {rtg.some_attr} {
%snippet = rtg.snippet attributes {rtg.some_attr} {
  %arg = arith.constant 1 : i32
  // CHECK: rtg.label "label_string_{0}_{1}", %{{.*}}, %{{.*}} : i32, i32
  rtg.label "label_string_{0}_{1}", %arg, %arg : i32, i32
  // CHECK: rtg.label "label_string"
  rtg.label "label_string"
}

// CHECK: rtg.snippet
rtg.snippet {
  // CHECK: [[RATIO:%.+]] = arith.constant 1 : i32
  %ratio = arith.constant 1 : i32
  // CHECK: rtg.select_random [[[SNIPPET]]], [[[RATIO]]]
  rtg.select_random [%snippet], [%ratio]
  // CHECK: rtg.select_random [[[SNIPPET]], [[SNIPPET]]], [[[RATIO]], [[RATIO]]]
  rtg.select_random [%snippet, %snippet], [%ratio, %ratio]
}

// CHECK-LABEL: @types
// CHECK-SAME: !rtg.snippet
// CHECK-SAME: !rtg.resource
func.func @types(%arg1: !rtg.snippet, %arg2: !rtg.resource) {
  return
}
