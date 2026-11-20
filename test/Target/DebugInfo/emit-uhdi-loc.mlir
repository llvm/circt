// RUN: circt-translate %s --emit-uhdi --uhdi-indent=0 --split-input-file | FileCheck %s

// A module is located at its source declaration in `original` and at its
// Verilog declaration in `generated`. Both index into `files`.

// CHECK:      "original":{"Module":{"loc":[0,2,1],
// CHECK-SAME: "generated":{"Module":{"loc":[1,1,8]}}
// CHECK-SAME: "files":[{"path":"Foo.scala"},{"path":"Foo.sv","generated":true}]
hw.module @Module() {
} loc(fused["Foo.scala":2:1, fused<"verilogLocations">["Foo.sv":1:8]])

// -----

// A variable is located at its declaration. A column of 0 is left out.

// CHECK:      "original":{"Vars":{
// CHECK-SAME: "a":{"type":0,"loc":[0,3,7]},
// CHECK-SAME: "b":{"type":0,"loc":[0,4]}
hw.module @Vars(in %a: i8) {
  dbg.variable "a", %a : i8 loc("Foo.scala":3:7)
  dbg.variable "b", %a : i8 loc("Foo.scala":4:0)
} loc("Foo.scala":1:1)

// -----

// An inlined module has no declaration left; only its instance is located.

// CHECK:      "original":{"Inlined":{
// CHECK-SAME: "s":{"type":0,"loc":[0,5,3]}
// CHECK-SAME: "S":{"vars":{}}
hw.module @Inlined() {
  %s = dbg.scope "s", "S" loc("Foo.scala":5:3)
} loc("Foo.scala":1:1)

// -----

// A signal is located at its Verilog declaration. A signal without one is
// left out, and of signals with the same name only the first is kept.

// CHECK: "generated":{"Signals":{"signals":{"w":{"loc":[1,2,3]}}}}
hw.module @Signals() {
  %w = sv.wire {hw.verilogName = "w"} : !hw.inout<i8> loc(fused["Foo.scala":6:3, fused<"verilogLocations">["Foo.sv":2:3]])
  %u = sv.wire {hw.verilogName = "u"} : !hw.inout<i8> loc("Foo.scala":7:3)
  %w2 = sv.wire {hw.verilogName = "w"} : !hw.inout<i8> loc(fused<"verilogLocations">["Foo.sv":9:3])
} loc("Foo.scala":1:1)
