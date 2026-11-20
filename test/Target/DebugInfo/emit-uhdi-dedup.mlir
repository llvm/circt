// RUN: circt-translate %s --emit-uhdi --uhdi-indent=0 --split-input-file | FileCheck %s

// DupA merged into Inl keeps its own variables and `dbg.module_locs` location,
// and is bound to the Verilog module Inl.

// CHECK:      "types":[{"kind":"uint","width":8}]
// CHECK-SAME: "original":{"Inl":{"loc":[0,2,1],"vars":{"st":{"type":0,"loc":[0,3,3]}}},
// CHECK-SAME: "DupA":{"loc":[0,10,1],"vars":{"alpha":{"type":0,"loc":[0,4,3]}}}
// CHECK-SAME: "generated":{"Inl":{}}
// CHECK-SAME: "bindings":[{"original":"Inl","generated":"Inl","vars":{"st":"st"}},
// CHECK-SAME: {"original":"DupA","generated":"Inl","vars":{"alpha":"st"}}]
hw.module private @Inl(in %a: i8) attributes {
    dbg.module_locs = {DupA = loc("D.scala":10:1), Inl = loc("D.scala":2:1)}} {
  %st = hw.wire %a : i8
  dbg.variable "st", %st : i8 loc("D.scala":3:3)
  dbg.variable "alpha", %st {dbg.merged_from = "DupA"} : i8 loc("D.scala":4:3)
} loc(fused["D.scala":2:1, "D.scala":10:1])

// -----

// An instance of a merged module keeps its own module type. A stale module
// name gives the target's type.

// CHECK:      "types":[{"kind":"uint","width":8},
// CHECK-SAME: {"kind":"module","module":"Inl"},
// CHECK-SAME: {"kind":"module","module":"DupA"}]
// CHECK-SAME: "Top":{
// CHECK-SAME: "i0":{"type":1,
// CHECK-SAME: "da":{"type":2,
// CHECK-SAME: "gone":{"type":1,
// CHECK-SAME: "generated":{"Inl":{},"Top":{}}
// CHECK-SAME: "i0":{"$inst":"i0","module":"Inl"},
// CHECK-SAME: "da":{"$inst":"da","module":"Inl"},
// CHECK-SAME: "gone":{"$inst":"gone","module":"Inl"}
hw.module private @Inl(in %a: i8) attributes {
    dbg.module_locs = {DupA = loc("D.scala":10:1), Inl = loc("D.scala":2:1)}} {
} loc("D.scala":2:1)

hw.module @Top(in %a: i8) {
  hw.instance "i0" @Inl(a: %a: i8) -> () {dbg.instance = [#dbg.instance<name = "i0">]}
  hw.instance "da" @Inl(a: %a: i8) -> () {dbg.instance = [#dbg.instance<name = "da", module = "DupA">]}
  hw.instance "gone" @Inl(a: %a: i8) -> () {dbg.instance = [#dbg.instance<name = "gone", module = "Gone">]}
} loc("D.scala":20:1)

// -----

// P2 merged into P1, and S2 into S1 before S1 was inlined into P1. Merged
// modules keep their own variables and bind to P1; inlined ones bind at their
// inline path. Instances of P1, P2 and a stale name get module types.

// CHECK:      "types":[{"kind":"uint","width":8},
// CHECK-SAME: {"kind":"module","module":"S1"},
// CHECK-SAME: {"kind":"module","module":"S2"},
// CHECK-SAME: {"kind":"module","module":"P1"},
// CHECK-SAME: {"kind":"module","module":"P2"}]
// CHECK-SAME: "original":{"P1":{"loc":[0,20,1],"vars":{
// CHECK-SAME: "p1var":{"type":0,
// CHECK-SAME: "s":{"type":1,
// CHECK-SAME: "P2":{"loc":[0,30,1],"vars":{
// CHECK-SAME: "p2var":{"type":0,
// CHECK-SAME: "s":{"type":2,
// CHECK-SAME: "Top":{
// CHECK-SAME: "i1":{"type":3,
// CHECK-SAME: "i2":{"type":4,
// CHECK-SAME: "i3":{"type":3,
// CHECK-SAME: "S1":{"vars":{"s1var":{"type":0,
// CHECK-SAME: "S2":{"vars":{"s2var":{"type":0,
// CHECK-SAME: "generated":{"P1":{},"Top":{}}
// CHECK-SAME: "bindings":[{"original":"P1","generated":"P1","vars":{"p1var":"p","s":{"$inline":["s"]}}},
// CHECK-SAME: {"original":"S1","generated":"P1","inline_path":["s"],"vars":{"s1var":"w"}},
// CHECK-SAME: {"original":"P2","generated":"P1","vars":{"p2var":"p","s":{"$inline":["s"]}}},
// CHECK-SAME: {"original":"S2","generated":"P1","inline_path":["s"],"vars":{"s2var":"w"}},
// CHECK-SAME: {"original":"Top"
hw.module private @P1(in %p: i8) attributes {
    dbg.module_locs = {P1 = loc("D.scala":20:1), P2 = loc("D.scala":30:1)}} {
  dbg.variable "p1var", %p : i8 loc("D.scala":21:3)
  dbg.variable "p2var", %p {dbg.merged_from = "P2"} : i8 loc("D.scala":22:3)
  %s = dbg.scope "s", "S1" {dbg.instance = [#dbg.instance<name = "s">, #dbg.instance<name = "s", module = "S2", origin = "P2">]} loc("D.scala":23:3)
  %w = hw.wire %p : i8
  dbg.variable "s1var", %w scope %s {dbg.merged_from = "S1"} : i8 loc("D.scala":25:3)
  dbg.variable "s2var", %w scope %s {dbg.merged_from = "S2"} : i8 loc("D.scala":26:3)
} loc("D.scala":20:1)

hw.module @Top(in %a: i8) {
  hw.instance "i1" @P1(p: %a: i8) -> () {dbg.instance = [#dbg.instance<name = "i1">]}
  hw.instance "i2" @P1(p: %a: i8) -> () {dbg.instance = [#dbg.instance<name = "i2", module = "P2">]}
  hw.instance "i3" @P1(p: %a: i8) -> () {dbg.instance = [#dbg.instance<name = "i3", module = "Gone">]}
} loc("D.scala":60:1)

// -----

// Q2 merged into Q1, and both inlined the same R. R is bound once.

// CHECK:      "types":[{"kind":"module","module":"R"},{"kind":"uint","width":8}]
// CHECK-SAME: "original":{"Q1":{"loc":[0,40,1],"vars":{"r":{"type":0,
// CHECK-SAME: "Q2":{"loc":[0,50,1],"vars":{"r":{"type":0,
// CHECK-SAME: "R":{"vars":{"rvar":{"type":1,
// CHECK-SAME: "generated":{"Q1":{}}
// CHECK-SAME: "bindings":[{"original":"Q1","generated":"Q1","vars":{"r":{"$inline":["r"]}}},
// CHECK-SAME: {"original":"R","generated":"Q1","inline_path":["r"],"vars":{"rvar":"q"}},
// CHECK-SAME: {"original":"Q2","generated":"Q1","vars":{"r":{"$inline":["r"]}}}]
hw.module private @Q1(in %q: i8) attributes {
    dbg.module_locs = {Q1 = loc("D.scala":40:1), Q2 = loc("D.scala":50:1)}} {
  %r = dbg.scope "r", "R" {dbg.instance = [#dbg.instance<name = "r">, #dbg.instance<name = "r", origin = "Q2">]} loc("D.scala":41:3)
  dbg.variable "rvar", %q scope %r : i8 loc("D.scala":43:3)
} loc("D.scala":40:1)
