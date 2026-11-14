// RUN: circt-translate %s --emit-uhdi --uhdi-indent=0 --split-input-file | FileCheck %s

// A packed aggregate is one signal. Its leaves are slices of that signal:
// the first struct field is the most significant, array element 0 the least.

// Type table: struct field a is uint<4>, b is a vector of two uint<3>.
// CHECK:      "types":[{"kind":"uint","width":4},{"kind":"uint","width":3},
// CHECK-SAME: {"elem":1,"kind":"vector","size":2},
// CHECK-SAME: {"fields":[{"name":"a","type":0},{"name":"b","type":2}],"kind":"struct"}]
// The port takes the struct type.
// CHECK-SAME: "original":{"Packed":{
// CHECK-SAME: "io":{"type":3,
// Leaves are slices of the port.
// CHECK-SAME: "bindings":[{"original":"Packed","generated":"Packed","vars":{
// CHECK-SAME: "io":{"a":{"$slice":"io","hi":9,"lo":6},
// CHECK-SAME: "b":[{"$slice":"io","hi":2,"lo":0},{"$slice":"io","hi":5,"lo":3}]}}
hw.module @Packed(in %io: !hw.struct<a: i4, b: !hw.array<2xi3>>) {
  %a = hw.struct_extract %io["a"] : !hw.struct<a: i4, b: !hw.array<2xi3>>
  %b = hw.struct_extract %io["b"] : !hw.struct<a: i4, b: !hw.array<2xi3>>
  %false = hw.constant false
  %true = hw.constant true
  %b0 = hw.array_get %b[%false] : !hw.array<2xi3>, i1
  %b1 = hw.array_get %b[%true] : !hw.array<2xi3>, i1
  %bs = dbg.array [%b0, %b1] : i3
  %s = dbg.struct {"a": %a, "b": %bs} : i4, !dbg.array
  dbg.variable "io", %s : !dbg.struct
} loc("Foo.scala":1:1)

// -----

// An external port of aggregate type gets its type from the port.

// CHECK: "original":{"Ext":{
// CHECK-SAME: "p":{"type":3,
hw.module.extern @Ext(in %p: !hw.struct<a: i4, b: !hw.array<2xi3>>) loc("Foo.scala":1:1)

// -----

// An index that is not a constant selects no fixed bits.

// CHECK: "original":{"Dyn":{
// CHECK-SAME: "dyn":{"$optimized_out":true}
hw.module @Dyn(in %b: !hw.array<2xi3>, in %i: i1) {
  %bi = hw.array_get %b[%i] : !hw.array<2xi3>, i1
  dbg.variable "dyn", %bi : i3
} loc("Foo.scala":1:1)

// -----

// A bit extract is a slice too, also of a slice.

// CHECK: "original":{"Extract":{
// CHECK-SAME: "ext":{"$slice":"w","hi":6,"lo":3},
// CHECK-SAME: "ext_a":{"$slice":"io","hi":8,"lo":7}
hw.module @Extract(in %io: !hw.struct<a: i4, b: !hw.array<2xi3>>, in %w: i9) {
  %a = hw.struct_extract %io["a"] : !hw.struct<a: i4, b: !hw.array<2xi3>>
  %ext = comb.extract %w from 3 : (i9) -> i4
  dbg.variable "ext", %ext : i4
  %ext_a = comb.extract %a from 1 : (i4) -> i2
  dbg.variable "ext_a", %ext_a : i2
} loc("Foo.scala":1:1)

// -----

// A field next to one of unknown width selects no fixed bits.

// CHECK: "original":{"Unsized":{
// CHECK-SAME: "unsized":{"$optimized_out":true}
hw.module @Unsized(in %u: !hw.struct<a: i4, s: !hw.string>) {
  %u_a = hw.struct_extract %u["a"] : !hw.struct<a: i4, s: !hw.string>
  dbg.variable "unsized", %u_a : i4
} loc("Foo.scala":1:1)
