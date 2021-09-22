// RUN: circt-opt %s  -test-schedulable-op-interface | FileCheck %s

// CHECK-LABEL: func @mac
func @mac(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
  // CHECK: %[[TMP0:.+]] = muli {{.+}} {opr = "three-cycle"}
  %0 = muli %arg0, %arg1 : i32
  // CHECK: %[[TMP1:.+]] = addi {{.+}} {opr = "comb"}
  %1 = addi %0, %arg2 : i32
  // CHECK: return %[[TMP1]] : i32
  return %1 : i32
}
