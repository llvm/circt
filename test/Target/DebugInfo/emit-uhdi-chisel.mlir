// RUN: circt-translate %s --emit-uhdi --uhdi-indent=0 --split-input-file | FileCheck %s

// The class of an instance, from its `#dbg.instance`, goes into `name` on its
// module type. Each `ext` entry becomes an `x-<lang>` key and a profile.

// CHECK:      "profiles":["x-chisel","x-other"]
// CHECK-SAME: "types":[{"kind":"module","module":"Child","name":"Child","x-chisel":{"params":[{"name":"b","value":"true"},{"name":"n","value":null}]},"x-other":[1]}]
hw.module private @Child() {
} loc("Foo.scala":1:1)

hw.module @Top() {
  hw.instance "c" @Child() -> () {dbg.instance = [#dbg.instance<name = "c", typeName = "Child", ext = {chisel = "{\"params\":[{\"name\":\"b\",\"value\":\"true\"},{\"name\":\"n\",\"value\":null}]}", other = "[1]"}>]}
} loc("Foo.scala":2:1)

// -----

// Instances of one module with the same class and `ext` share a type. An
// instance without a class gets a plain type.

// CHECK:      "types":[{"kind":"module","module":"Child","name":"Child","x-chisel":{"n":1}},
// CHECK-SAME: {"kind":"module","module":"Child"}
// CHECK-SAME: "Top":{
// CHECK-SAME: "c":{"type":0,
// CHECK-SAME: "c2":{"type":1,
// CHECK-SAME: "c3":{"type":0,
hw.module private @Child() {
} loc("Foo.scala":1:1)

hw.module @Top() {
  hw.instance "c" @Child() -> () {dbg.instance = [#dbg.instance<name = "c", typeName = "Child", ext = {chisel = "{\"n\":1}"}>]}
  hw.instance "c2" @Child() -> ()
  hw.instance "c3" @Child() -> () {dbg.instance = [#dbg.instance<name = "c3", typeName = "Child", ext = {chisel = "{\"n\":1}"}>]}
} loc("Foo.scala":2:1)

// -----

// The class of a public module, from `dbg.moduleinfo`, goes on the module.

// CHECK:      "original":{"Top":{
// CHECK-SAME: "name":"Top","x-chisel":{"params":[]},"vars":{
hw.module @Top(in %a: i8) attributes {dbg.moduleinfo = {name = "Top", ext = {chisel = "{\"params\":[]}"}}} {
  dbg.variable "a", %a : i8
} loc("Foo.scala":1:1)

// -----

// An inlined scope with a class gets a named module type. Its module has
// no declaration and so no `name`.

// CHECK:      "types":[{"kind":"module","module":"Inl","name":"Inl"}]
// CHECK-SAME: "i":{"type":0,
// CHECK-SAME: "Inl":{"vars":{}}
// CHECK-SAME: "bindings":[
hw.module @Top() {
  %i = dbg.scope "i", "Inl" typeName "Inl"
} loc("Foo.scala":1:1)

// -----

// After dedup each merged module reads the class of an instance off its own
// scope, for a kept instance and for an inlined one.

// CHECK:      "types":[
// CHECK-SAME: {"kind":"module","module":"S1","name":"ClsS1"},
// CHECK-SAME: {"kind":"module","module":"T1","name":"ClsT1"},
// CHECK-SAME: {"kind":"module","module":"S2","name":"ClsS2"},
// CHECK-SAME: {"kind":"module","module":"T2","name":"ClsT2"}
// CHECK-SAME: "P1":{
// CHECK-SAME: "s":{"type":1,
// CHECK-SAME: "t":{"type":2,
// CHECK-SAME: "P2":{
// CHECK-SAME: "s":{"type":3,
// CHECK-SAME: "t":{"type":4,
hw.module private @S1(in %x: i8) attributes {
    dbg.module_locs = {S1 = loc("S.scala":1:1), S2 = loc("S.scala":2:1)}} {
  dbg.variable "x", %x : i8
  dbg.variable "x", %x {dbg.merged_from = "S2"} : i8
} loc("S.scala":1:1)

hw.module private @P1(in %x: i8) attributes {
    dbg.module_locs = {P1 = loc("P.scala":1:1), P2 = loc("P.scala":2:1)}} {
  hw.instance "s" @S1(x: %x: i8) -> () {dbg.instance = [#dbg.instance<name = "s", typeName = "ClsS1">, #dbg.instance<name = "s", module = "S2", typeName = "ClsS2", origin = "P2">]}
  %t = dbg.scope "t", "T1" typeName "ClsT1" {dbg.instance = [#dbg.instance<name = "t", typeName = "ClsT1">, #dbg.instance<name = "t", module = "T2", typeName = "ClsT2", origin = "P2">]}
  dbg.variable "y", %x scope %t {dbg.merged_from = "T1"} : i8
  dbg.variable "y", %x scope %t {dbg.merged_from = "T2"} : i8
} loc("P.scala":1:1)

hw.module @Dedup(in %x: i8) {
  hw.instance "p1" @P1(x: %x: i8) -> ()
} loc("P.scala":3:1)
