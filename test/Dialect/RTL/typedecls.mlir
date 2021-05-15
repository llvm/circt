// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: hw.type_scope @__hw_typedecls {
hw.type_scope @__hw_typedecls {
  // CHECK: hw.typedecl @foo : i1
  hw.typedecl @foo : i1
  // CHECK: hw.typedecl @bar : !hw.struct<a: i1, b: i1>
  hw.typedecl @bar : !hw.struct<a: i1, b: i1>
  // CHECK: hw.typedecl @baz, "MY_NAMESPACE_baz" : i8
  hw.typedecl @baz, "MY_NAMESPACE_baz" : i8
}
