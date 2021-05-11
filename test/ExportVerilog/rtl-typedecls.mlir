// RUN: circt-translate %s -export-verilog -verify-diagnostics | FileCheck %s --strict-whitespace

rtl.type_scope @__rtl_typedecls {
  // CHECK: typedef logic foo;
  rtl.typedecl @foo : i1
  // CHECK: typedef struct packed {logic a; logic b; } bar;
  rtl.typedecl @bar : !rtl.struct<a: i1, b: i1>
  // CHECK: typedef logic [7:0][0:15] baz;
  rtl.typedecl @baz : !rtl.uarray<16xi8>
}
