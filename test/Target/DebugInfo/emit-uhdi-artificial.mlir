// RUN: circt-translate %s --emit-uhdi --uhdi-indent=0 --split-input-file | FileCheck %s

// A module the compiler made, such as one a layer block is extracted into, is
// artificial. Its variables are not.

// CHECK:      "original":{"Art":{"loc":[0,1,1],"artificial":true,
// CHECK-SAME: "vars":{"check":{"type":0,"loc":[0,2,3]}}}
hw.module private @Art(in %x: i8) attributes {dbg.artificial} {
  dbg.variable "check", %x : i8 loc("Foo.scala":2:3)
} loc("Foo.scala":1:1)

// -----

// A source module is not artificial, nor are its variables.

// CHECK:      "original":{"Plain":{"loc":[0,1,1],"vars":{"a":{"type":0,"loc":[0,2,3]}}}
hw.module @Plain(in %a: i8) {
  dbg.variable "a", %a : i8 loc("Foo.scala":2:3)
} loc("Foo.scala":1:1)

// -----

// The instance of an artificial module is an artificial variable. It has no
// name in the source and keeps its Verilog name, which cannot clash with a
// source variable.

// CHECK:      "original":{"Top_A":{"loc":[0,1,1],"artificial":true,
// CHECK-SAME: "vars":{"x":{"type":0,"loc":[0,1,1],"dir":"in"}}},
// CHECK-SAME: "Top":{"loc":[0,4,1],"vars":{"a":{"type":0,"loc":[0,5,3]},
// CHECK-SAME: "a_0":{"type":1,"loc":[0,6,3],"artificial":true}}}
// CHECK-SAME: "bindings":[
// CHECK-SAME: {"original":"Top","generated":"Top","vars":{"a":"a",
// CHECK-SAME: "a_0":{"$inst":"a_0","module":"Top_A"}}}
// CHECK-SAME: "files":[{"path":"Foo.scala"}]
hw.module private @Top_A(in %x: i8 loc("Foo.scala":1:1)) attributes {dbg.artificial} {
} loc("Foo.scala":1:1)

hw.module @Top(in %a: i8) {
  dbg.variable "a", %a : i8 loc("Foo.scala":5:3)
  hw.instance "a" @Top_A(x: %a: i8) -> () {doNotPrint, hw.verilogName = "a_0"} loc("Foo.scala":6:3)
} loc("Foo.scala":4:1)
