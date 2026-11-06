// RUN: circt-translate %s --emit-uhdi --uhdi-indent=0 --split-input-file | FileCheck %s

// A variable key is unique in its module and has none of `.`, `[`, `]`. A key
// that differs from the declared name is accompanied by `name`.

// CHECK:      "original":{"Foo":{
// CHECK-SAME: "x":{"type":0,
// CHECK-SAME: "x_1":{"name":"x",
// CHECK-SAME: "a_b_c_":{"name":"a.b[c]",
// CHECK-SAME: "a_b_c__1":{"name":"a_b_c_",
// CHECK-SAME: "bindings":[{"original":"Foo","generated":"Foo","vars":{"x":"a","x_1":"b","a_b_c_":"a","a_b_c__1":"b"}}
hw.module @Foo(in %a : i8, in %b : i8) {
  dbg.variable "x", %a : i8
  dbg.variable "x", %b : i8
  dbg.variable "a.b[c]", %a : i8
  dbg.variable "a_b_c_", %b : i8
} loc("Foo.scala":1:1)

// -----

// Every inlined copy of a module binds all of its variables.

// CHECK:      "original":{"Twice":{
// CHECK-SAME: "bindings":[
// CHECK-SAME: {"original":"Inl","generated":"Twice","inline_path":["i1"],"vars":{"st":"a"}}
// CHECK-SAME: {"original":"Inl","generated":"Twice","inline_path":["i2"],"vars":{"st":"b"}}
hw.module @Twice(in %a : i8, in %b : i8) {
  %i1 = dbg.scope "i1", "Inl"
  dbg.variable "st", %a scope %i1 : i8
  %i2 = dbg.scope "i2", "Inl"
  dbg.variable "st", %b scope %i2 : i8
} loc("Foo.scala":1:1)

// -----

// A copy that lists its variables in another order
// still gets the keys of the module in `original`.

// CHECK:      "original":{"TwiceSwapped":{
// CHECK-SAME: "Swap":{"vars":{
// CHECK-SAME: "a_b":{"name":"a.b",
// CHECK-SAME: "a_b_1":{"name":"a_b",
// CHECK-SAME: {"original":"Swap","generated":"TwiceSwapped","inline_path":["j1"],"vars":{"a_b":"a","a_b_1":"b"}}
// CHECK-SAME: {"original":"Swap","generated":"TwiceSwapped","inline_path":["j2"],"vars":{"a_b_1":"b","a_b":"a"}}
hw.module @TwiceSwapped(in %a : i8, in %b : i8) {
  %j1 = dbg.scope "j1", "Swap"
  dbg.variable "a.b", %a scope %j1 : i8
  dbg.variable "a_b", %b scope %j1 : i8
  %j2 = dbg.scope "j2", "Swap"
  dbg.variable "a_b", %b scope %j2 : i8
  dbg.variable "a.b", %a scope %j2 : i8
} loc("Foo.scala":1:1)
