// RUN: circt-translate %s -export-verilog -verify-diagnostics | FileCheck %s --strict-whitespace

rtl.type_scope @__rtl_typedecls {
  // CHECK-LABEL: typedef logic foo;
  rtl.typedecl @foo : i1
  // CHECK-LABEL: typedef struct packed {logic a; logic b; } bar;
  rtl.typedecl @bar : !rtl.struct<a: i1, b: i1>
  // CHECK-LABEL: typedef logic [7:0][0:15] baz;
  rtl.typedecl @baz : !rtl.uarray<16xi8>
  // CHECK-LABEL: typedef logic [31:0] customName;
  rtl.typedecl @qux, "customName" : i32
}

// CHECK-LABEL: module testTypeRef
rtl.module @testTypeRef(
  // CHECK: input foo        arg0
  %arg0: !rtl.typeref<@__rtl_typedecls::@foo>,
  // CHECK: input customName arg1
  %arg1: !rtl.typeref<@__rtl_typedecls::@qux>) {
}
