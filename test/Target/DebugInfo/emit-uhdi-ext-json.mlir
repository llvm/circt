// RUN: circt-translate %s --emit-uhdi --uhdi-indent=0 --split-input-file | FileCheck %s

// `ext` is parsed and written again with its keys sorted. Two values whose
// `ext` differs only in key order share one type.

// CHECK:      "types":[{"kind":"uint","width":1,"x-chisel":{"a":3,"b":1}}]
// CHECK-SAME: "Sorted":{
// CHECK-SAME: "e1":{"type":0
// CHECK-SAME: "e2":{"type":0
hw.module @Sorted(in %x : i1, in %y : i1) {
  %e1 = dbg.value %x ext {chisel = "{\"b\": 1, \"a\": 3}"} : i1 loc("Foo.scala":1:1)
  dbg.variable "e1", %e1 : !dbg.value loc("Foo.scala":1:1)
  %e2 = dbg.value %y ext {chisel = "{\"a\": 3, \"b\": 1}"} : i1 loc("Foo.scala":1:1)
  dbg.variable "e2", %e2 : !dbg.value loc("Foo.scala":1:1)
} loc("Foo.scala":1:1)

// -----

// A duplicate key in `ext` keeps its last value.

// CHECK:      "types":[{"kind":"uint","width":1,"x-chisel":{"a":3,"b":1}}]
// CHECK-SAME: "Duplicate":{
// CHECK-SAME: "e":{"type":0
hw.module @Duplicate(in %x : i1) {
  %e = dbg.value %x ext {chisel = "{\"a\": 2, \"b\": 1, \"a\": 3}"} : i1 loc("Foo.scala":1:1)
  dbg.variable "e", %e : !dbg.value loc("Foo.scala":1:1)
} loc("Foo.scala":1:1)
