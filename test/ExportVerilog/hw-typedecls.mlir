// RUN: circt-translate %s -export-verilog -verify-diagnostics | FileCheck %s --strict-whitespace

hw.type_scope @__hw_typedecls {
  // CHECK: typedef logic foo;
  hw.typedecl @foo : i1
  // CHECK: typedef struct packed {logic a; logic b; } bar;
  hw.typedecl @bar : !hw.struct<a: i1, b: i1>
  // CHECK: typedef logic [7:0][0:15] baz;
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
  // CHECK: input  foo arg0, arg1
  %arg0: !hw.typealias<@__hw_typedecls::@foo,i1>,
  %arg1: !hw.typealias<@__hw_typedecls::@foo,i1>,
  // CHECK: input  arr arrArg,
  %arrArg: !hw.typealias<@__hw_typedecls::@arr,!hw.array<16xi8>>,
  // CHECK: input  bar structArg,
  %structArg: !hw.typealias<@__hw_typedecls::@bar,!hw.struct<a: i1, b: i1>>) ->
  // CHECK: output foo out
  (%out: !hw.typealias<@__hw_typedecls::@foo, i1>) {
  // CHECK: out = arg0 + arg1
  %0 = comb.add %arg0, %arg1 : !hw.typealias<@__hw_typedecls::@foo, i1>
  hw.output %0 : !hw.typealias<@__hw_typedecls::@foo, i1>
}
