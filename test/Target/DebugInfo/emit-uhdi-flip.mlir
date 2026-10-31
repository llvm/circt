// RUN: circt-translate %s --emit-uhdi --uhdi-indent=0 --split-input-file | FileCheck %s

// A flipped field gets `"flip": true`.

// CHECK:      "types":[{"kind":"uint","width":1},
// CHECK-SAME: {"fields":[{"name":"x","type":0},{"flip":true,"name":"y","type":0}],"kind":"struct"}]
// CHECK-SAME: "original":{"Flipped":{
// CHECK-SAME: "flipped":{"type":1,
hw.module @Flipped(in %x: i1, in %y: i1) {
  %f = dbg.struct {"x": %x, "y": %y} {flips = array<i1: false, true>} : i1, i1
  dbg.variable "flipped", %f dir out : !dbg.struct loc("Foo.scala":2:3)
} loc("Foo.scala":1:1)

// -----

// The flip is part of the type: two structs that differ only in a flip are
// two types.

// CHECK:      "types":[{"kind":"uint","width":1},
// CHECK-SAME: {"fields":[{"name":"x","type":0},{"flip":true,"name":"y","type":0}],"kind":"struct"},
// CHECK-SAME: {"fields":[{"name":"x","type":0},{"name":"y","type":0}],"kind":"struct"}]
// CHECK-SAME: "original":{"Plain":{
// CHECK-SAME: "flipped":{"type":1,
// CHECK-SAME: "plain":{"type":2,
hw.module @Plain(in %x: i1, in %y: i1) {
  %f = dbg.struct {"x": %x, "y": %y} {flips = array<i1: false, true>} : i1, i1
  dbg.variable "flipped", %f dir out : !dbg.struct loc("Foo.scala":2:3)
  %p = dbg.struct {"x": %x, "y": %y} : i1, i1
  dbg.variable "plain", %p dir out : !dbg.struct loc("Foo.scala":3:3)
} loc("Foo.scala":1:1)

// -----

// A flip inside a flipped field keeps its own flag.

// CHECK:      "types":[{"kind":"uint","width":1},
// CHECK-SAME: {"fields":[{"name":"x","type":0},{"flip":true,"name":"y","type":0}],"kind":"struct"},
// CHECK-SAME: {"fields":[{"flip":true,"name":"a","type":1},{"name":"x","type":0}],"kind":"struct"}]
// CHECK-SAME: "original":{"Nested":{
// CHECK-SAME: "nested":{"type":2,
hw.module @Nested(in %x: i1, in %y: i1) {
  %f = dbg.struct {"x": %x, "y": %y} {flips = array<i1: false, true>} : i1, i1
  %n = dbg.struct {"a": %f, "x": %x} {flips = array<i1: true, false>} : !dbg.struct, i1
  dbg.variable "nested", %n dir out : !dbg.struct loc("Foo.scala":2:3)
} loc("Foo.scala":1:1)
