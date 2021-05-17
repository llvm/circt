// RUN: circt-translate %s -export-verilog -verify-diagnostics | FileCheck %s --strict-whitespace

hw.type_scope @__hw_typedecls {
  // CHECK: typedef logic foo;
  hw.typedecl @foo : i1
  // CHECK: typedef struct packed {logic a; logic b; } bar;
  hw.typedecl @bar : !hw.struct<a: i1, b: i1>
  // CHECK: typedef logic [7:0][0:15] baz;
  hw.typedecl @baz : !hw.uarray<16xi8>
}
