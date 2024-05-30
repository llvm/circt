// RUN: circt-opt %s --verify-diagnostics | circt-opt | FileCheck %s

// CHECK-LABEL: func.func @Foo
func.func @Foo(%arg0: i32, %arg1: index, %arg2: f64) {
  // CHECK-NEXT: dbg.variable "foo", %arg0 : i32
  // CHECK-NEXT: dbg.variable "bar", %arg1 : index
  // CHECK-NEXT: dbg.variable "baz", %arg2 : f64
  dbg.variable "foo", %arg0 : i32
  dbg.variable "bar", %arg1 : index
  dbg.variable "baz", %arg2 : f64

  // CHECK-NEXT: [[TMP:%.+]] = dbg.struct {"foo": %arg0, "bar": %arg1, "baz": %arg2} : i32, index, f64
  // CHECK-NEXT: dbg.variable "megafoo", [[TMP]] : !dbg.struct
  %0 = dbg.struct {"foo": %arg0, "bar": %arg1, "baz": %arg2} : i32, index, f64
  dbg.variable "megafoo", %0 : !dbg.struct

  // CHECK-NEXT: [[TMP:%.+]] = dbg.array [%arg1, %arg1] : index
  // CHECK-NEXT: dbg.variable "megabar", [[TMP]] : !dbg.array
  %1 = dbg.array [%arg1, %arg1] : index
  dbg.variable "megabar", %1 : !dbg.array

  // CHECK-NEXT: [[TMP:%.+]] = dbg.scope "inlined", "Bar"
  // CHECK-NEXT: dbg.variable {{.+}} scope [[TMP]]
  // CHECK-NEXT: dbg.scope {{.+}} scope [[TMP]]
  %2 = dbg.scope "inlined", "Bar"
  dbg.variable "x", %arg0 scope %2 : i32
  dbg.scope "y", "Baz" scope %2

  // CHECK-NEXT: [[SUBFIELD0:%.+]] = dbg.subfield "subfield0", %arg0 : i32
  // CHECK-NEXT: [[SUBFIELD1:%.+]] = dbg.subfield "subfield1", %arg1 : index
  // CHECK-NEXT: [[SUBFIELD2:%.+]] = dbg.subfield "subfield2", %arg2 : f64
  %subfield0 = dbg.subfield "subfield0",  %arg0 : i32
  %subfield1 = dbg.subfield "subfield1",  %arg1 : index
  %subfield2 = dbg.subfield "subfield2",  %arg2 : f64

  // CHECK-NEXT: [[TMP:%.+]] = dbg.struct {"subfield0": [[SUBFIELD0]], "subfield1": [[SUBFIELD1]], "subfield2": [[SUBFIELD2]]} : !dbg.subfield, !dbg.subfield, !dbg.subfield
  // CHECK-NEXT: [[FOO:%.+]] = dbg.subfield "megasubfieldFoo", [[TMP]] : !dbg.struct
  %megasubfieldFoo = dbg.struct {"subfield0": %subfield0, "subfield1": %subfield1, "subfield2": %subfield2} : !dbg.subfield, !dbg.subfield, !dbg.subfield
  %3 = dbg.subfield "megasubfieldFoo", %megasubfieldFoo : !dbg.struct

  // CHECK-NEXT: [[TMP:%.+]] = dbg.array [[[SUBFIELD1]], [[SUBFIELD1]]] : !dbg.subfield
  // CHECK-NEXT: [[BAR:%.+]] = dbg.subfield "megasubfieldBar", [[TMP]] : !dbg.array
  %megasubfieldBar = dbg.array [%subfield1, %subfield1] : !dbg.subfield
  %4 = dbg.subfield "megasubfieldBar", %megasubfieldBar : !dbg.array
  
  // CHECK-NEXT: [[TMP:%.+]] = dbg.array [[[FOO]], [[BAR]]] : !dbg.subfield
  // CHECK-NEXT: dbg.variable "megaVar", [[TMP]] : !dbg.array
  %5 = dbg.array [%3, %4] : !dbg.subfield
  dbg.variable "megaVar", %5 : !dbg.array

  return
}

