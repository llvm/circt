// RUN: circt-opt %s -test-component-library | FileCheck %s

// CHECK-LABEL: func @mac
func @mac(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
  // CHECK: %[[TMP0:.+]] = muli {{.+}} {latency = 3 : i64}
  %0 = muli %arg0, %arg1 : i32
  // CHECK: %[[TMP1:.+]] = addi {{.+}} {latency = 0 : i64}
  %1 = addi %0, %arg2 : i32
  // CHECK: return %[[TMP1]] : i32
  return %1 : i32
}
