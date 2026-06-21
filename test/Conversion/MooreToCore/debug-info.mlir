// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// CHECK-LABEL: func.func @DebugValueAndEnum
// CHECK-SAME: (%arg0: i2)
func.func @DebugValueAndEnum(%arg0: !moore.i2) {
  // CHECK: %[[ENUM:.+]] = dbg.enum %arg0, "State", {Idle = 0 : i64, Run = 1 : i64} fqn "pkg.State" : i2
  %enum = dbg.enum %arg0, "State", {Idle = 0 : i64, Run = 1 : i64} fqn "pkg.State" : !moore.i2

  // CHECK: %[[VALUE:.+]] = dbg.value %[[ENUM]] typeName "State" : !dbg.enum
  %value = dbg.value %enum typeName "State" : !dbg.enum

  // CHECK: dbg.variable "state", %[[VALUE]] : !dbg.value
  dbg.variable "state", %value : !dbg.value
  return
}
