// RUN: circt-opt %s --verify-roundtrip | FileCheck %s
//
// Parser/printer round-trip for ops introduced alongside source-language
// type tracking: dbg.enum embedded in a dbg.variable.
// Aggregate/scope shapes are covered in basic.mlir.

// CHECK-LABEL: func.func @EnumAndVariable
func.func @EnumAndVariable() {
  %c = arith.constant 0 : i2
  // CHECK: %[[E:.+]] = dbg.enum %{{.*}}, "MyState", {Idle = 0 : i64, Run = 1 : i64} fqn "pkg.MyState$" : i2
  %e = dbg.enum %c, "MyState", {Idle = 0 : i64, Run = 1 : i64} fqn "pkg.MyState$" : i2
  // CHECK: %[[V:.+]] = dbg.value %[[E]] typeName "MyState" : !dbg.enum
  %v = dbg.value %e typeName "MyState" : !dbg.enum
  // CHECK: dbg.variable "state", %[[V]] : !dbg.value
  dbg.variable "state", %v : !dbg.value
  return
}

// CHECK-LABEL: func.func @EnumNoFqn
func.func @EnumNoFqn() {
  %c = arith.constant 0 : i2
  // CHECK: dbg.enum %{{.*}}, "S", {A = 0 : i64} : i2
  %e = dbg.enum %c, "S", {A = 0 : i64} : i2
  dbg.variable "s", %e : !dbg.enum
  return
}
