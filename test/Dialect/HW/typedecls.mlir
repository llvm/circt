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
  // CHECK: hw.typedecl @fooInt : i4
  hw.typedecl @fooInt : i4
  // CHECK: hw.typedecl @fooNested : !hw.typealias<@__hw_typedecls::@fooInt, i4>
  hw.typedecl @fooNested : !hw.typealias<@__hw_typedecls::@fooInt, i4>
}

// CHECK-LABEL: hw.module.extern @testTypeAlias
hw.module.extern @testTypeAlias(
  // CHECK: !hw.typealias<@__hw_typedecls::@foo, i1>
  in %arg0: !hw.typealias<@__hw_typedecls::@foo, i1>,
  // CHECK: !hw.typealias<@__hw_typedecls::@nested, !hw.struct<a: !hw.typealias<@__hw_typedecls::@foo, i1>, b: i1>>
  in %arg1: !hw.typealias<@__hw_typedecls::@nested, !hw.struct<a: !hw.typealias<@__hw_typedecls::@foo, i1>, b: i1>>
)

// CHECK-LABEL: hw.module @testTypeAliasComb
hw.module @testTypeAliasComb(
  in %arg0: !hw.typealias<@__hw_typedecls::@foo, i1>,
  in %arg1: !hw.typealias<@__hw_typedecls::@foo, i1>,
  out result: !hw.typealias<@__hw_typedecls::@foo, i1>) {
  // CHECK: comb.add %arg0, %arg1 : !hw.typealias<@__hw_typedecls::@foo, i1>
  %0 = comb.add %arg0, %arg1 : !hw.typealias<@__hw_typedecls::@foo, i1>
  hw.output %0 : !hw.typealias<@__hw_typedecls::@foo, i1>
}

// CHECK-LABEL: hw.module @testTypeAliasArray
!Foo = !hw.typealias<@__hw_typedecls::@foo, i1>
!FooArray = !hw.typealias<@__hw_typedecls::@fooArray, !hw.array<2x!Foo>>
hw.module @testTypeAliasArray(in %arg0: !Foo, in %arg1: !Foo, in %arg2: !FooArray) {
  %c1 = hw.constant 1 : i1
  %0 = hw.array_create %arg0, %arg1 : !Foo
  %1 = hw.array_create %arg0, %arg1 : !Foo -> !FooArray
  %2 = hw.array_concat %arg2, %arg2 : !FooArray, !FooArray
  %3 = hw.array_slice %arg2[%c1] : (!FooArray) -> !hw.array<1x!Foo>
  %4 = hw.array_get %arg2[%c1] : !FooArray, i1
  %5 = hw.aggregate_constant [false, true] : !FooArray
}

// CHECK-LABEL: hw.module @testTypeAliasStruct
!FooStruct = !hw.typealias<@__hw_typedecls::@fooStruct, !hw.struct<a: i1>>
hw.module @testTypeAliasStruct(in %arg0: !FooStruct, in %arg1: i1) {
  %0 = hw.struct_extract %arg0["a"] : !FooStruct
  %1 = hw.struct_inject %arg0["a"], %arg1 : !FooStruct
  %2:1 = hw.struct_explode %arg0 : !FooStruct
  %3 = hw.aggregate_constant [false] : !FooStruct
}

// CHECK-LABEL: hw.module @testTypeAliasUnion
!FooUnion = !hw.typealias<@__hw_typedecls::@fooUnion, !hw.union<a: i1>>
hw.module @testTypeAliasUnion(in %arg0: !FooUnion, in %arg1: i1) {
  %0 = hw.union_extract %arg0["a"] : !FooUnion
}

// CHECK-LABEL: hw.module @testTypeAliasConstant
!FooInt = !hw.typealias<@__hw_typedecls::@fooInt, i4>
!FooBool = !hw.typealias<@__hw_typedecls::@foo, i1>
!FooNested = !hw.typealias<@__hw_typedecls::@fooNested, !FooInt>
hw.module @testTypeAliasConstant(out a: !FooInt, out b: !FooInt, out c: !FooBool, out d: !FooNested, out e: i4) {
  // CHECK: %c5_fooInt = hw.constant 5 : !hw.typealias<@__hw_typedecls::@fooInt, i4>
  %0 = hw.constant 5 : !FooInt
  // CHECK: %c-1_fooInt = hw.constant -1 : !hw.typealias<@__hw_typedecls::@fooInt, i4>
  %1 = hw.constant 15 : !FooInt
  // CHECK: %true = hw.constant true : !hw.typealias<@__hw_typedecls::@foo, i1>
  %2 = hw.constant true : !FooBool
  // CHECK: %c3_fooNested = hw.constant 3 : !hw.typealias<@__hw_typedecls::@fooNested, !hw.typealias<@__hw_typedecls::@fooInt, i4>>
  %3 = hw.constant 3 : !FooNested
  // CHECK: %c3_i4 = hw.constant 3 : i4{{$}}
  %4 = hw.constant 3 : i4
  hw.output %0, %1, %2, %3, %4 : !FooInt, !FooInt, !FooBool, !FooNested, i4
}
