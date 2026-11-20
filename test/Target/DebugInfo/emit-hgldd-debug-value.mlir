// RUN: circt-translate %s --emit-hgldd | FileCheck %s

// A `dbg.value` or `dbg.enum` wrapper is metadata: a wrapped variable gets the
// same value as a plain one.

#loc1 = loc("Foo.scala":4:10)

// CHECK-LABEL: "obj_name": "Foo"
// CHECK:       "var_name": "plain"
// CHECK:       "value": {"sig_name":"a"}
// CHECK-NEXT:  "type_name": "logic"
// CHECK:       "var_name": "wrapped"
// CHECK:       "value": {"sig_name":"a"}
// CHECK-NEXT:  "type_name": "logic"
// CHECK:       "var_name": "state"
// CHECK:       "value": {"sig_name":"a"}
// CHECK-NEXT:  "type_name": "logic"
// CHECK:       "var_name": "aggr"
// CHECK:       "value": {"opcode":"'{","operands":[{"sig_name":"a"},{"sig_name":"a"}]}
// CHECK-NEXT:  "type_name": "Foo_aggr"
// CHECK:       "var_name": "expr"
// CHECK:       "value": {"opcode":"+","operands":[{"sig_name":"a"},{"sig_name":"a"}]}
// CHECK-NEXT:  "type_name": "logic"
hw.module @Foo(in %a: i8, out c: i8) {
  dbg.variable "plain", %a : i8 loc(#loc1)
  %w = dbg.value %a typeName "IO[UInt<8>]" : i8 loc(#loc1)
  dbg.variable "wrapped", %w : !dbg.value loc(#loc1)
  %e = dbg.enum %a, "S", {A = 0 : i64} : i8 loc(#loc1)
  dbg.variable "state", %e : !dbg.enum loc(#loc1)
  %s = dbg.struct {"x": %w, "y": %w} : !dbg.value, !dbg.value loc(#loc1)
  dbg.variable "aggr", %s : !dbg.struct loc(#loc1)
  %sum = comb.add %a, %a : i8 loc(#loc1)
  %ws = dbg.value %sum : i8 loc(#loc1)
  dbg.variable "expr", %ws : !dbg.value loc(#loc1)
  hw.output %a : i8
} loc(#loc1)
