// RUN: circt-translate %s --emit-uhdi --uhdi-indent=0 --split-input-file | FileCheck %s

// A variable of a type that cannot be shown is left out of `original` and of
// the bindings. Only the type of the other variable stays in the type table.

// CHECK:      "types":[{"kind":"uint","width":8}],"original":{"Str":{
// CHECK-SAME: "vars":{"ok":{"type":0,"loc":[0,2,1]}}}}
// CHECK-SAME: "bindings":[{"original":"Str","generated":"Str","vars":{"ok":"a"}}]
hw.module @Str(in %a : i8, in %s : !hw.string) {
  dbg.variable "str", %s : !hw.string
  dbg.variable "ok", %a : i8 loc("Foo.scala":2:1)
} loc("Foo.scala":1:1)

// -----

// The same holds when the unshowable type is nested. The array and struct
// types built on the way to it are dropped too.

// CHECK:      "types":[{"kind":"uint","width":8}],"original":{"Mixed":{
// CHECK-SAME: "vars":{"ok":{"type":0,"loc":[0,2,1]}}}}
// CHECK-SAME: "bindings":[{"original":"Mixed","generated":"Mixed","vars":{"ok":"a"}}]
hw.module @Mixed(in %a : i8, in %e : !hw.array<2xstruct<p: i2, q: !hw.string>>) {
  dbg.variable "mixed", %e : !hw.array<2xstruct<p: i2, q: !hw.string>>
  dbg.variable "ok", %a : i8 loc("Foo.scala":2:1)
} loc("Foo.scala":1:1)

// -----

// An empty array has no element to give its type, so its variable is left out.

// CHECK:      "types":[{"kind":"uint","width":8}],"original":{"Empty":{
// CHECK-SAME: "vars":{"ok":{"type":0,"loc":[0,5,1]}}}}
// CHECK-SAME: "bindings":[{"original":"Empty","generated":"Empty","vars":{"ok":"a"}}]
hw.module @Empty(in %a : i8) {
  %none = dbg.array []
  %empty = dbg.struct {"none": %none} : !dbg.array
  dbg.variable "empty", %empty : !dbg.struct
  dbg.variable "ok", %a : i8 loc("Foo.scala":5:1)
} loc("Foo.scala":1:1)
