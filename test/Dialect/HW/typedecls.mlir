// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: hw.type_scope @__hw_typedecls {
hw.type_scope @__hw_typedecls {
  // CHECK: hw.typedecl @foo : i1
  hw.typedecl @foo : i1
  // CHECK: hw.typedecl @bar : !hw.struct<a: i1, b: i1>
  hw.typedecl @bar : !hw.struct<a: i1, b: i1>
  // CHECK: hw.typedecl @baz, "MY_NAMESPACE_baz" : i8
  hw.typedecl @baz, "MY_NAMESPACE_baz" : i8
  // CHECK: hw.typedecl @nested : !hw.struct<a: !hw.typealias<@__hw_typedecls::@foo, i1>, b: i1>
  hw.typedecl @nested : !hw.struct<a: !hw.typealias<@__hw_typedecls::@foo, i1>, b: i1>
}

// CHECK-LABEL: hw.module.extern @testTypeAlias
hw.module.extern @testTypeAlias(
  // CHECK: !hw.typealias<@__hw_typedecls::@foo, i1>
  %arg0: !hw.typealias<@__hw_typedecls::@foo, i1>,
  // CHECK: !hw.typealias<@__hw_typedecls::@nested, !hw.struct<a: !hw.typealias<@__hw_typedecls::@foo, i1>, b: i1>>
  %arg1: !hw.typealias<@__hw_typedecls::@nested, !hw.struct<a: !hw.typealias<@__hw_typedecls::@foo, i1>, b: i1>>
)

// CHECK-LABEL: hw.module @testTypeAliasComb
hw.module @testTypeAliasComb(
  %arg0: !hw.typealias<@__hw_typedecls::@foo, i1>,
  %arg1: !hw.typealias<@__hw_typedecls::@foo, i1>) -> (!hw.typealias<@__hw_typedecls::@foo, i1>) {
  // CHECK: comb.add %arg0, %arg1 : !hw.typealias<@__hw_typedecls::@foo, i1>
  %0 = comb.add %arg0, %arg1 : !hw.typealias<@__hw_typedecls::@foo, i1>
  hw.output %0 : !hw.typealias<@__hw_typedecls::@foo, i1>
}

// CHECK-LABEL: hw.module.extern @testTypeAlias
hw.module.extern @testTypeAlias(
  // CHECK: hw.typealias<@__hw_typedecls::@foo, i1>
  %arg0: !hw.typealias<@__hw_typedecls::@foo, i1>
)

// CHECK-LABEL: hw.module @testTypeAliasComb
hw.module @testTypeAliasComb(
  %arg0: !hw.typealias<@__hw_typedecls::@foo, i1>,
  %arg1: !hw.typealias<@__hw_typedecls::@foo, i1>) -> (!hw.typealias<@__hw_typedecls::@foo, i1>) {
  // CHECK: comb.add %arg0, %arg1 : !hw.typealias<@__hw_typedecls::@foo, i1>
  %0 = comb.add %arg0, %arg1 : !hw.typealias<@__hw_typedecls::@foo, i1>
  hw.output %0 : !hw.typealias<@__hw_typedecls::@foo, i1>
}
