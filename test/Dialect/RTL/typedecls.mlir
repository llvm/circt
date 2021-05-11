// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: rtl.type_scope @__rtl_typedecls {
rtl.type_scope @__rtl_typedecls {
  // CHECK: rtl.typedecl @foo : i1
  rtl.typedecl @foo : i1
  // CHECK: rtl.typedecl @bar : !rtl.struct<a: i1, b: i1>
  rtl.typedecl @bar : !rtl.struct<a: i1, b: i1>
  // CHECK: rtl.typedecl @baz, "MY_NAMESPACE_baz" : i8
  rtl.typedecl @baz, "MY_NAMESPACE_baz" : i8
}
