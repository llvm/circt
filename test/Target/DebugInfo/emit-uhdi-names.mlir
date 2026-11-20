// RUN: circt-translate %s --emit-uhdi --uhdi-indent=0 --split-input-file | FileCheck %s

// The name of a `dbg.value` goes on its type. A plain variable has an unnamed
// type, and a named and an unnamed scalar of one width are two types.

// CHECK:      "types":[{"kind":"uint","width":8},{"kind":"uint","name":"Bool","width":1},{"kind":"uint","width":1}]
// CHECK-SAME: "original":{"Named":{
// CHECK-SAME: "a":{"type":0,
// CHECK-SAME: "b":{"type":1,
// CHECK-SAME: "c":{"type":2,
hw.module @Named(in %a: i8, in %b: i1) {
  dbg.variable "a", %a : i8
  %f = dbg.value %b typeName "Bool" : i1
  dbg.variable "b", %f : !dbg.value
  dbg.variable "c", %b : i1
} loc("Foo.scala":1:1)

// -----

// The language data of a `dbg.value` goes on its type, with the JSON as it is.

// CHECK:      "types":[{"kind":"uint","width":8},{"kind":"uint","name":"Bool","width":1},
// CHECK-SAME: {"fields":[{"name":"x","type":0},{"name":"y","type":1}],"kind":"struct","name":"P","x-chisel":{"p":[true,null,"true",[1,[2]]]}}]
// CHECK-SAME: "original":{"Ext":{
// CHECK-SAME: "p":{"type":2,
hw.module @Ext(in %a: i8, in %b: i1) {
  %f = dbg.value %b typeName "Bool" : i1
  %s1 = dbg.struct {"x": %a, "y": %f} : i8, !dbg.value
  %p = dbg.value %s1 typeName "P" ext {chisel = "{\"p\":[true,null,\"true\",[1,[2]]]}"} : !dbg.struct
  dbg.variable "p", %p : !dbg.value
} loc("Foo.scala":1:1)

// -----

// Bundles of one shape and different names, or with and without a name, are
// different types. Bundles with the same name and language data are one type.

// CHECK:      "types":[{"kind":"uint","width":8},{"kind":"uint","name":"Bool","width":1},
// CHECK-SAME: {"fields":[{"name":"x","type":0},{"name":"y","type":1}],"kind":"struct","name":"P","x-chisel":{"p":[true,null,"true",[1,[2]]]}},
// CHECK-SAME: {"fields":[{"name":"x","type":0},{"name":"y","type":1}],"kind":"struct","name":"Q"},
// CHECK-SAME: {"fields":[{"name":"x","type":0},{"name":"y","type":1}],"kind":"struct"}]
// CHECK-SAME: "original":{"Shapes":{
// CHECK-SAME: "p":{"type":2,
// CHECK-SAME: "q":{"type":3,
// CHECK-SAME: "r":{"type":4,
// CHECK-SAME: "p2":{"type":2,
hw.module @Shapes(in %a: i8, in %b: i1) {
  %f = dbg.value %b typeName "Bool" : i1
  %s1 = dbg.struct {"x": %a, "y": %f} : i8, !dbg.value
  %p = dbg.value %s1 typeName "P" ext {chisel = "{\"p\":[true,null,\"true\",[1,[2]]]}"} : !dbg.struct
  dbg.variable "p", %p : !dbg.value
  %s2 = dbg.struct {"x": %a, "y": %f} : i8, !dbg.value
  %q = dbg.value %s2 typeName "Q" : !dbg.struct
  dbg.variable "q", %q : !dbg.value
  %s3 = dbg.struct {"x": %a, "y": %f} : i8, !dbg.value
  dbg.variable "r", %s3 : !dbg.struct
  %s4 = dbg.struct {"x": %a, "y": %f} : i8, !dbg.value
  %p2 = dbg.value %s4 typeName "P" ext {chisel = "{\"p\":[true,null,\"true\",[1,[2]]]}"} : !dbg.struct
  dbg.variable "p2", %p2 : !dbg.value
} loc("Foo.scala":1:1)

// -----

// A type made only to be named is not left in the table: the unsigned `i4`
// under the `sint` value is gone.

// CHECK:      "types":[{"kind":"sint","name":"SInt<4>","width":4}]
// CHECK-SAME: "original":{"Signed":{
// CHECK-SAME: "k":{"type":0,
hw.module @Signed(in %s: i4) {
  %k = dbg.value %s typeName "SInt<4>" sint : i4
  dbg.variable "k", %k : !dbg.value
} loc("Foo.scala":1:1)

// -----

// A `dbg.enum` carries its language data on the enum type.

// CHECK:      "types":[{"kind":"uint","width":1},
// CHECK-SAME: {"base":0,"kind":"enum","name":"S","values":{"A":0},"x-chisel":null}]
// CHECK-SAME: "original":{"Enum":{
// CHECK-SAME: "e":{"type":1,
hw.module @Enum(in %c: i1) {
  %e = dbg.enum %c, "S", {A = 0 : i64} ext {chisel = "null"} : i1
  dbg.variable "e", %e : !dbg.enum
} loc("Foo.scala":1:1)

// -----

// An array wrapped in a `dbg.value` carries its language data on the vector
// type.

// CHECK:      "types":[{"kind":"uint","width":1},
// CHECK-SAME: {"elem":0,"kind":"vector","size":2,"x-chisel":[]}]
// CHECK-SAME: "original":{"Vector":{
// CHECK-SAME: "v":{"type":1,
hw.module @Vector(in %b: i1) {
  %v = dbg.array [%b, %b] : i1
  %vn = dbg.value %v ext {chisel = "[]"} : !dbg.array
  dbg.variable "v", %vn : !dbg.value
} loc("Foo.scala":1:1)
