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

  return
}

