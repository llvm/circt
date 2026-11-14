// RUN: circt-translate %s --emit-uhdi --uhdi-indent=0 --split-input-file | FileCheck %s

// A `dbg.value` wrapper without a name or language data is metadata: a wrapped
// variable gets the same type and value as a plain one.

// CHECK:      "types":[{"kind":"uint","width":8}]
// CHECK-SAME: "original":{"Plain":{
// CHECK-SAME: "plain":{"type":0,
// CHECK-SAME: "wrapped":{"type":0,
// CHECK-SAME: "bindings":[{
// CHECK-SAME: "vars":{"plain":"a","wrapped":"a"}
hw.module @Plain(in %a: i8) {
  dbg.variable "plain", %a : i8
  %w = dbg.value %a : i8
  dbg.variable "wrapped", %w : !dbg.value
} loc("a.mlir":1:1)

// -----

// A wrapped value used as a field of an aggregate resolves to the same signal.

// CHECK:      "types":[{"kind":"uint","width":8},{"fields":[{"name":"x","type":0}],"kind":"struct"}]
// CHECK-SAME: "original":{"Aggr":{
// CHECK-SAME: "aggr":{"type":1,
// CHECK-SAME: "bindings":[{
// CHECK-SAME: "vars":{"aggr":{"x":"a"}}
hw.module @Aggr(in %a: i8) {
  %w = dbg.value %a : i8
  %s = dbg.struct {"x": %w} : !dbg.value
  dbg.variable "aggr", %s : !dbg.struct
} loc("a.mlir":1:1)
