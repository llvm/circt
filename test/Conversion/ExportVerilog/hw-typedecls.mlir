// RUN: circt-opt %s -export-verilog -verify-diagnostics | FileCheck %s --strict-whitespace

hw.type_scope @__hw_typedecls {
  // CHECK: typedef logic foo;
  hw.typedecl @foo : i1
  // CHECK: typedef struct packed {logic a; logic b; } bar;
  hw.typedecl @bar : !hw.struct<a: i1, b: i1>

  // CHECK: typedef struct packed {logic a; logic [7:0] b[0:15]; } barArray;
  hw.typedecl @barArray : !hw.struct<a: i1, b: !hw.uarray<16xi8>>

  // CHECK: typedef logic [7:0] baz[0:15];
  hw.typedecl @baz : !hw.uarray<16xi8>
  // CHECK: typedef logic [15:0][7:0] arr;
  hw.typedecl @arr : !hw.array<16xi8>
  // CHECK: typedef logic [31:0] customName;
  hw.typedecl @qux, "customName" : i32
  // CHECK: typedef struct packed {foo a; _other_scope_foo b; } nestedRef;
  hw.typedecl @nestedRef : !hw.struct<a: !hw.typealias<@__hw_typedecls::@foo,i1>, b: !hw.typealias<@_other_scope::@foo,i2>>
}

hw.type_scope @_other_scope {
  // CHECK: typedef logic [1:0] _other_scope_foo;
  hw.typedecl @foo, "_other_scope_foo" : i2
}

// CHECK-LABEL: module testTypeAlias
hw.module @testTypeAlias(
  // CHECK: input  foo      arg0, arg1
  %arg0: !hw.typealias<@__hw_typedecls::@foo,i1>,
  %arg1: !hw.typealias<@__hw_typedecls::@foo,i1>,
  // CHECK: input  foo[2:0] arg2
  %arg2: !hw.array<3xtypealias<@__hw_typedecls::@foo,i1>>,
  // CHECK: input  arr      arrArg,
  %arrArg: !hw.typealias<@__hw_typedecls::@arr,!hw.array<16xi8>>,
  // CHECK: input  bar      structArg,
  %structArg: !hw.typealias<@__hw_typedecls::@bar,!hw.struct<a: i1, b: i1>>) ->
  // CHECK: output foo      out
  (out: !hw.typealias<@__hw_typedecls::@foo, i1>) {
  // CHECK: out = arg0 + arg1
  %0 = comb.add %arg0, %arg1 : !hw.typealias<@__hw_typedecls::@foo, i1>
  hw.output %0 : !hw.typealias<@__hw_typedecls::@foo, i1>
}

// CHECK-LABEL: module testRegOp
hw.module @testRegOp() -> () {
  // CHECK: foo {{.+}};
  %r1 = sv.reg : !hw.inout<!hw.typealias<@__hw_typedecls::@foo,i1>>
  // CHECK: foo[2:0] {{.+}};
  %r2 = sv.reg : !hw.inout<!hw.array<3xtypealias<@__hw_typedecls::@foo,i1>>>
}

// CHECK-LABEL: module testAggregateCreate
hw.module @testAggregateCreate(%i: i1) -> (out1: i1, out2: i1) {
  // CHECK: wire bar [[NAME:.+]] = {{.+}};
  %0 = hw.struct_create(%i, %i) : !hw.typealias<@__hw_typedecls::@bar,!hw.struct<a: i1, b: i1>>
  // CHECK: [[NAME]].a
  %1 = hw.struct_extract %0["a"] : !hw.typealias<@__hw_typedecls::@bar,!hw.struct<a: i1, b: i1>>
  // CHECK: [[NAME]].b
  %2 = hw.struct_extract %0["b"] : !hw.typealias<@__hw_typedecls::@bar,!hw.struct<a: i1, b: i1>>
  hw.output %1, %2 : i1, i1
}
