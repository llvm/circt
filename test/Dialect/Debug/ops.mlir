// RUN: circt-opt %s --verify-roundtrip | FileCheck %s
//
// Parser/printer round-trip for ops introduced alongside source-language
// type tracking: dbg.enum embedded in a dbg.variable,
// register variables, port directions, value kinds and struct flips.
// Aggregate/scope shapes are covered in basic.mlir.

// CHECK-LABEL: func.func @EnumAndVariable
func.func @EnumAndVariable() {
  %c = arith.constant 0 : i2
  // CHECK: %[[E:.+]] = dbg.enum %{{.*}}, "MyState", {Idle = 0 : i64, Run = 1 : i64} : i2
  %e = dbg.enum %c, "MyState", {Idle = 0 : i64, Run = 1 : i64} : i2
  // CHECK: %[[V:.+]] = dbg.value %[[E]] typeName "MyState" : !dbg.enum
  %v = dbg.value %e typeName "MyState" : !dbg.enum
  // CHECK: dbg.variable "state", %[[V]] : !dbg.value
  dbg.variable "state", %v : !dbg.value
  return
}

// CHECK-LABEL: func.func @Ext
func.func @Ext(%a: i8) {
  %c = arith.constant 0 : i2
  // CHECK: %[[E:.+]] = dbg.enum %{{.*}}, "S", {A = 0 : i64} ext {chisel = "[1]"} : i2
  %e = dbg.enum %c, "S", {A = 0 : i64} ext {chisel = "[1]"} : i2
  // CHECK: dbg.variable "s", %[[E]] : !dbg.enum
  dbg.variable "s", %e : !dbg.enum
  // CHECK: %[[V:.+]] = dbg.value %arg0 typeName "SInt<8>" ext {chisel = "null"} sint : i8
  %v = dbg.value %a typeName "SInt<8>" ext {chisel = "null"} sint : i8
  // CHECK: dbg.variable "v", %[[V]] : !dbg.value
  dbg.variable "v", %v : !dbg.value
  return
}

// CHECK-LABEL: func.func @VariableKind
func.func @VariableKind(%a: i8) {
  // CHECK: dbg.variable "w", %arg0 wire : i8
  dbg.variable "w", %a wire : i8
  // CHECK: dbg.variable "r", %arg0 reg : i8
  dbg.variable "r", %a reg : i8
  // CHECK: dbg.variable "m", %arg0 mem : i8
  dbg.variable "m", %a mem : i8
  // CHECK: dbg.variable "n", %arg0 node : i8
  dbg.variable "n", %a node : i8
  return
}

// CHECK-LABEL: func.func @ValueKind
func.func @ValueKind(%a: i1, %b: i8) {
  // CHECK: %[[R:.+]] = dbg.value %arg0 reset : i1
  // CHECK: dbg.variable "rst", %[[R]] : !dbg.value
  %r = dbg.value %a reset : i1
  dbg.variable "rst", %r : !dbg.value
  // CHECK: %[[C:.+]] = dbg.value %arg0 clock : i1
  // CHECK: dbg.variable "clk", %[[C]] : !dbg.value
  %c = dbg.value %a clock : i1
  dbg.variable "clk", %c : !dbg.value
  // CHECK: %[[N:.+]] = dbg.value %arg1 sint : i8
  // CHECK: dbg.variable "neg", %[[N]] : !dbg.value
  %n = dbg.value %b sint : i8
  dbg.variable "neg", %n : !dbg.value
  // CHECK: %[[A:.+]] = dbg.value %arg0 asyncreset : i1
  // CHECK: dbg.variable "arst", %[[A]] reg : !dbg.value
  %s = dbg.value %a asyncreset : i1
  dbg.variable "arst", %s reg : !dbg.value
  return
}

// CHECK-LABEL: func.func @PortDir
func.func @PortDir(%a: i8) {
  // CHECK: dbg.variable "i", %arg0 dir in : i8
  dbg.variable "i", %a dir in : i8
  // CHECK: dbg.variable "o", %arg0 dir out : i8
  dbg.variable "o", %a dir out : i8
  // CHECK: dbg.variable "io", %arg0 dir inout : i8
  dbg.variable "io", %a dir inout : i8
  // CHECK: dbg.variable "q", %arg0 reg dir out : i8
  dbg.variable "q", %a reg dir out : i8
  return
}

// CHECK-LABEL: func.func @StructFlips
func.func @StructFlips(%a: i1, %b: i8) {
  // CHECK: dbg.struct {"x": %{{.*}}, "y": %{{.*}}} {flips = array<i1: false, true>} : i1, i8
  %s = dbg.struct {"x": %a, "y": %b} {flips = array<i1: false, true>} : i1, i8
  dbg.variable "s", %s : !dbg.struct
  return
}

hw.module @Alu() {}

// CHECK-LABEL: hw.module @Instance
hw.module @Instance() {
  // CHECK: hw.instance "a" @Alu() -> () {dbg.instance = [#dbg.instance<name = "a">]}
  hw.instance "a" @Alu() -> () {dbg.instance = [#dbg.instance<name = "a">]}
  // CHECK: hw.instance "b" @Alu() -> () {dbg.instance = [#dbg.instance<name = "b", typeName = "Alu", ext = {chisel = "[8]"}>]}
  hw.instance "b" @Alu() -> () {dbg.instance = [#dbg.instance<name = "b", typeName = "Alu", ext = {chisel = "[8]"}>]}
  // CHECK: %[[S:.+]] = dbg.scope "c", "Inl" typeName "Inl"
  %0 = dbg.scope "c", "Inl" typeName "Inl"
  // CHECK: hw.instance "c_d" @Alu() -> () {dbg.instance = [#dbg.instance<name = "d", path = ["c"]>]}
  hw.instance "c_d" @Alu() -> () {dbg.instance = [#dbg.instance<name = "d", path = ["c"]>]}
  // CHECK: hw.instance "i" @Alu() -> () {dbg.instance = [#dbg.instance<name = "i">, #dbg.instance<name = "j", module = "Alu2", origin = "Instance2">]}
  hw.instance "i" @Alu() -> () {dbg.instance = [#dbg.instance<name = "i">, #dbg.instance<name = "j", module = "Alu2", origin = "Instance2">]}
  // CHECK: dbg.scope "g", "Inl" scope %[[S]] typeName "Inl" ext {chisel = "[1]"}
  %1 = dbg.scope "g", "Inl" scope %0 typeName "Inl" ext {chisel = "[1]"}
  // CHECK: hw.instance "c_g_h" @Alu() -> () {dbg.instance = [#dbg.instance<name = "h", typeName = "Alu", path = ["c", "g"]>]}
  hw.instance "c_g_h" @Alu() -> () {dbg.instance = [#dbg.instance<name = "h", typeName = "Alu", path = ["c", "g"]>]}
}
