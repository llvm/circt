// RUN: circt-translate %s --emit-uhdi --uhdi-indent=0 --split-input-file | FileCheck %s

// A register variable has `storage`, a node has `node` and a port has `dir`.
// A wire and a plain variable have none of them.

// CHECK:      "types":[{"kind":"uint","width":8}]
// CHECK-SAME: "original":{"Attrs":{
// CHECK-SAME: "r":{"type":0,"loc":[0,2,1],"storage":"reg"},
// CHECK-SAME: "w":{"type":0,"loc":[0,3,1]},
// CHECK-SAME: "n":{"type":0,"loc":[0,4,1],"node":true},
// CHECK-SAME: "d":{"type":0,"loc":[0,5,1],"dir":"in"},
// CHECK-SAME: "x":{"type":0,"loc":[0,6,1]}
hw.module @Attrs(in %d: i8) {
  dbg.variable "r", %d reg : i8 loc("Foo.scala":2:1)
  dbg.variable "w", %d : i8 loc("Foo.scala":3:1)
  dbg.variable "n", %d node : i8 loc("Foo.scala":4:1)
  dbg.variable "d", %d dir in : i8 loc("Foo.scala":5:1)
  dbg.variable "x", %d wire : i8 loc("Foo.scala":6:1)
} loc("Foo.scala":1:1)

// -----

// A clock, a reset or a signed integer gets its own type kind instead of
// `uint`. A plain i1 stays `uint`.

// CHECK:      "types":[{"kind":"asyncreset"},{"kind":"reset"},{"kind":"uint","width":1},{"kind":"clock"},{"kind":"sint","width":8}]
// CHECK-SAME: "original":{"Kinds":{
// CHECK-SAME: "arst":{"type":0,
// CHECK-SAME: "srst":{"type":1,
// CHECK-SAME: "bit":{"type":2,
// CHECK-SAME: "clk":{"type":3,
// CHECK-SAME: "neg":{"type":4,
// CHECK-SAME: "bindings":[{"original":"Kinds","generated":"Kinds","vars":{"arst":"a","srst":"s","bit":"s","clk":"s","neg":"d"}}]
hw.module @Kinds(in %a: i1, in %s: i1, in %d: i8) {
  %0 = dbg.value %a asyncreset : i1
  dbg.variable "arst", %0 : !dbg.value loc("Foo.scala":2:1)
  %1 = dbg.value %s reset : i1
  dbg.variable "srst", %1 : !dbg.value loc("Foo.scala":3:1)
  dbg.variable "bit", %s : i1 loc("Foo.scala":4:1)
  %2 = dbg.value %s clock : i1
  dbg.variable "clk", %2 : !dbg.value loc("Foo.scala":5:1)
  %3 = dbg.value %d sint : i8
  dbg.variable "neg", %3 : !dbg.value loc("Foo.scala":6:1)
} loc("Foo.scala":1:1)

// -----

// An inout port has `dir` "inout"; an input port has `dir` "in".

// CHECK:      "original":{"Pad":{
// CHECK-SAME: "p":{"type":0,"loc":[0,2,3],"dir":"inout"},
// CHECK-SAME: "q":{"type":0,"loc":[0,3,3],"dir":"in"}
hw.module.extern @Pad(in %p : !hw.inout<i1> loc("Foo.scala":2:3), in %q : i1 loc("Foo.scala":3:3)) loc("Foo.scala":1:1)
