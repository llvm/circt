// RUN: circt-opt %s --split-input-file -mlir-print-debuginfo \
// RUN:   --strip-debuginfo-with-pred=drop-suffix=mlir \
// RUN:   | circt-translate --split-input-file --emit-uhdi --uhdi-indent=0 | FileCheck %s

// The document starts with its version, producer and profiles.

// CHECK: {"uhdi":"0.1","producer":"CIRCT {{[^"]*}}","profiles":[],
hw.module @Header() {
} loc("a.mlir":1:1)

// -----

// An inout wire has the type of its element.

// CHECK:      "types":[{"kind":"uint","width":8}],"original":{"Inout":{"vars":{"w":{"type":0}}}}
// CHECK-SAME: "bindings":[{"original":"Inout","generated":"Inout","vars":{"w":"w_1"}}]
hw.module @Inout() {
  %w = sv.wire {hw.verilogName = "w_1"} : !hw.inout<i8>
  dbg.variable "w", %w : !hw.inout<i8>
} loc("a.mlir":1:1)

// -----

// Variables bind to the Verilog names of ports, which may differ from their IR
// names. `b` reads the input port it is driven from.

// CHECK:      "original":{"Names":{"vars":{"a":{"type":0},"b":{"type":0}}}}
// CHECK-SAME: "bindings":[{"original":"Names","generated":"Names","vars":{"a":"a_0","b":"a_0"}}]
hw.module @Names(in %a: i8 {hw.verilogName = "a_0"}, out b: i8) {
  dbg.variable "a", %a : i8
  dbg.variable "b", %a : i8
  hw.output %a : i8
} loc("a.mlir":1:1)

// -----

// The Verilog name of a module can differ from its name in the IR. An instance
// kept in Verilog names its instance and Verilog module.

// CHECK:      "original":{"Child":{"vars":{}},"Bar":{"vars":{"x":{"type":0},"child":{"type":1}}}}
// CHECK-SAME: "generated":{"Child":{},"Bar_v":{}}
// CHECK-SAME: {"original":"Bar","generated":"Bar_v","vars":{"x":"x","child":{"$inst":"child","module":"Child"}}}]
hw.module @Child() {
} loc("a.mlir":1:1)

hw.module @Bar(in %x: i8) attributes {verilogName = "Bar_v"} {
  dbg.variable "x", %x : i8
  hw.instance "child" @Child() -> ()
} loc("a.mlir":1:1)

// -----

// An external module takes its variables and their locations from its ports.

// CHECK:      "original":{"Ext":{"vars":{
// CHECK-SAME: "clk":{"type":0,"loc":[0,2,9],"dir":"in"},
// CHECK-SAME: "q":{"type":1,"loc":[0,3,15],"dir":"out"}
// CHECK-SAME: "generated":{"Ext":{"kind":"extern"}}
// CHECK-SAME: "bindings":[{"original":"Ext","generated":"Ext","vars":{"clk":"clk","q":"q"}}]
// CHECK-SAME: "files":[{"path":"Ext.scala"}]
hw.module.extern @Ext(in %clk: i1 loc("Ext.scala":2:9), out q: i8 loc("Ext.scala":3:15)) loc("a.mlir":1:1)

// -----

// Equal types share one entry, and a type comes after the types it is built
// from.

// CHECK:      "types":[{"kind":"clock"},
// CHECK-SAME: {"kind":"uint","width":8},
// CHECK-SAME: {"kind":"uint","width":2},
// CHECK-SAME: {"elem":2,"kind":"vector","size":2},
// CHECK-SAME: {"fields":[{"name":"a","type":1},{"name":"b","type":1},{"name":"v","type":3}],"kind":"struct"},
// CHECK-SAME: {"base":2,"kind":"enum","name":"Op","values":{"Add":0,"Sub":1}}],
// CHECK-SAME: "original":{"AggTypes":{"vars":{"clk":{"type":0},"s":{"type":4},"op":{"type":5}}}}
hw.module @AggTypes(in %clk: !seq.clock, in %a: i8, in %b: i8, in %v0: i2, in %v1: i2, in %op: i2) {
  dbg.variable "clk", %clk : !seq.clock
  %v = dbg.array [%v0, %v1] : i2
  %s = dbg.struct {"a": %a, "b": %b, "v": %v} : i8, i8, !dbg.array
  dbg.variable "s", %s : !dbg.struct
  %e = dbg.enum %op, "Op", {Add = 0 : i2, Sub = 1 : i2} : i2
  dbg.variable "op", %e : !dbg.enum
} loc("a.mlir":1:1)

// -----

// A tag that fits 64 bits is a JSON number, even above the int64 range.

// CHECK: "values":{"Big":9223372036854775808}
hw.module @WideEnum(in %a: i64) {
  %e = dbg.enum %a, "W", {Big = 9223372036854775808 : i64} : i64
  dbg.variable "w", %e : !dbg.enum
} loc("a.mlir":1:1)

// -----

// An aggregate value has the shape of its type; an enum is its tag.

// CHECK:      "bindings":[{"original":"Agg","generated":"Agg","vars":{
// CHECK-SAME: "clk":"clk",
// CHECK-SAME: "s":{"a":"a","b":"b","v":["v0","v1"]},
// CHECK-SAME: "op":"op"}}]
hw.module @Agg(in %clk: !seq.clock, in %a: i8, in %b: i8, in %v0: i2, in %v1: i2, in %op: i2) {
  dbg.variable "clk", %clk : !seq.clock
  %v = dbg.array [%v0, %v1] : i2
  %s = dbg.struct {"a": %a, "b": %b, "v": %v} : i8, i8, !dbg.array
  dbg.variable "s", %s : !dbg.struct
  %e = dbg.enum %op, "Op", {Add = 0 : i2, Sub = 1 : i2} : i2
  dbg.variable "op", %e : !dbg.enum
} loc("a.mlir":1:1)

// -----

// Taken from `firtool -g` output: the wire is read through `sv.read_inout`.

// CHECK:      "original":{"Leaf":{"vars":{"x":{"type":0},"y":{"type":0},"t":{"type":1}}}}
// CHECK-SAME: "bindings":[{"original":"Leaf","generated":"Leaf","vars":{"x":"x","y":"y","t":"t"}}]
hw.module private @Leaf(in %x : i8, out y : i8) {
  %c1_i9 = hw.constant 1 : i9
  %false = hw.constant false
  %0 = comb.concat %false, %x : i1, i8
  %1 = comb.add bin %0, %c1_i9 : i9
  %t = sv.wire {hw.verilogName = "t"} : !hw.inout<i9>
  sv.assign %t, %1 : i9
  %2 = sv.read_inout %t : !hw.inout<i9>
  %3 = sv.read_inout %t : !hw.inout<i9>
  %4 = comb.extract %3 from 0 {sv.namehint = ".y.output"} : (i9) -> i8
  dbg.variable "x", %x : i8
  dbg.variable "y", %4 : i8
  dbg.variable "t", %2 : i9
  hw.output %4 : i8
} loc("a.mlir":1:1)

// -----

// A zero-width value has a type of width 0.

// CHECK:      {"kind":"uint","width":0}
// CHECK-SAME: "original":{"Zero":{"vars":{"none":{"type":0}}}}
hw.module @Zero() {
  %c0_i0 = hw.constant 0 : i0
  dbg.variable "none", %c0_i0 : i0
} loc("a.mlir":1:1)

// -----

// A literal is `$const` with all bits of its width; a zero-width value is
// `$zero_width`. A value that no signal carries is `$optimized_out`.

// CHECK:      "bindings":[{"original":"Lit","generated":"Lit","vars":{
// CHECK-SAME: "k":{"$const":"0b0101"},
// CHECK-SAME: "sum":{"$optimized_out":true},
// CHECK-SAME: "op":{"$const":"0b01"},
// CHECK-SAME: "none":{"$zero_width":true},
// CHECK-SAME: "p":{"k":{"$const":"0b0101"},"sum":{"$optimized_out":true}}}}]
hw.module @Lit(in %a: i8, in %b: i8) {
  %c5_i4 = hw.constant 5 : i4
  dbg.variable "k", %c5_i4 : i4
  %sum = comb.add %a, %b : i8
  dbg.variable "sum", %sum : i8
  %c1_i2 = hw.constant 1 : i2
  %e = dbg.enum %c1_i2, "Op", {Add = 0 : i2, Sub = 1 : i2} : i2
  dbg.variable "op", %e : !dbg.enum
  %c0_i0 = hw.constant 0 : i0
  dbg.variable "none", %c0_i0 : i0
  %p = dbg.struct {"k": %c5_i4, "sum": %sum} : i4, i8
  dbg.variable "p", %p : !dbg.struct
} loc("a.mlir":1:1)

// -----

// An inlined module is an original module of its own; its instance is a
// variable of module type.

// CHECK:      {"kind":"module","module":"B"}
// CHECK-SAME: {"kind":"module","module":"C"}
// CHECK-SAME: "original":{"OuterTypes":{"vars":{"b":{"type":0}}},
// CHECK-SAME: "B":{"vars":{"x":{"type":1},"c":{"type":2}}},
// CHECK-SAME: "C":{"vars":{"y":{"type":1}}}}
hw.module @OuterTypes(in %a : i8) {
  %b = dbg.scope "b", "B"
  dbg.variable "x", %a scope %b : i8
  %c = dbg.scope "c", "C" scope %b
  dbg.variable "y", %a scope %c : i8
} loc("a.mlir":1:1)

// -----

// An inlined instance gives the `inline_path` of the binding that holds its
// variables. Each level of nested inlining gets its own binding.

// CHECK:      "bindings":[{"original":"Outer","generated":"Outer","vars":{"b":{"$inline":["b"]}}},
// CHECK-SAME: {"original":"B","generated":"Outer","inline_path":["b"],"vars":{"x":"a","c":{"$inline":["b","c"]}}},
// CHECK-SAME: {"original":"C","generated":"Outer","inline_path":["b","c"],"vars":{"y":"a"}}]
hw.module @Outer(in %a : i8) {
  %b = dbg.scope "b", "B"
  dbg.variable "x", %a scope %b : i8
  %c = dbg.scope "c", "C" scope %b
  dbg.variable "y", %a scope %c : i8
} loc("a.mlir":1:1)
