// RUN: circt-opt %s --verify-roundtrip | FileCheck %s
//
// Parser/printer round-trip for ops introduced alongside source-language
// type tracking: dbg.moduleinfo, dbg.enumdef, and dbg.variable's enumDef
// operand. Aggregate/scope shapes are covered in basic.mlir.

// CHECK-LABEL: func.func @ModuleInfoPlain
func.func @ModuleInfoPlain() {
  // CHECK: dbg.moduleinfo typeName "MyMod"
  dbg.moduleinfo typeName "MyMod"
  return
}

// CHECK-LABEL: func.func @ModuleInfoWithParams
func.func @ModuleInfoWithParams() {
  // CHECK: dbg.moduleinfo typeName "MyClass" params [{name = "arg0", value = "42"}]
  dbg.moduleinfo typeName "MyClass" params [{name = "arg0", value = "42"}]
  return
}

// CHECK-LABEL: func.func @EnumDefAndVariable
func.func @EnumDefAndVariable() {
  // CHECK: %[[E:.+]] = dbg.enumdef "MyState", fqn "pkg.MyState$", {Idle = 0 : i64, Run = 1 : i64}
  %e = dbg.enumdef "MyState", fqn "pkg.MyState$", {Idle = 0 : i64, Run = 1 : i64}
  %c = arith.constant 0 : i2
  // CHECK: dbg.variable "state", %{{.*}} typeName "MyState" enumDef %[[E]] : i2
  dbg.variable "state", %c typeName "MyState" enumDef %e : i2
  return
}

// CHECK-LABEL: func.func @EnumDefWithScope
func.func @EnumDefWithScope() {
  // CHECK: %[[S:.+]] = dbg.scope "inner", "InnerMod"
  %scope = dbg.scope "inner", "InnerMod"
  // CHECK: dbg.enumdef "S", fqn "p.S", {A = 0 : i64} scope %[[S]]
  %e = dbg.enumdef "S", fqn "p.S", {A = 0 : i64} scope %scope
  return
}
