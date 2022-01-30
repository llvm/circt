// RUN: circt-opt -pass-pipeline='lower-firrtl-to-hw{create-type-declarations=true}' -verify-diagnostics %s | FileCheck %s
firrtl.circuit "UseFoo" {
  // Global type scope.
  // CHECK-LABEL:   hw.type_scope
  // CHECK-SAME:     @[[GLOBAL_TYPE_SCOPE:.+]] {
  // CHECK-NEXT:      hw.typedecl @[[TYPE1:.+]], "Global_a" : !hw.struct<b: i1>
  // CHECK-NEXT:      hw.typedecl @[[TYPE2:.+]] : !hw.struct<a: !hw.typealias<@[[GLOBAL_TYPE_SCOPE]]::@[[TYPE1]], !hw.struct<b: i1>>>
  // CHECK-NEXT:      hw.typedecl @[[TYPE3:.+]] : !hw.struct<ext: i1>
  // CHECK-NEXT:    }

  // CHECK-LABEL:   hw.module @Foo
  // CHECK-SAME:     (%in: i1) -> (sink: !hw.typealias<@[[GLOBAL_TYPE_SCOPE]]::@[[TYPE2]], !hw.struct<a: !hw.typealias<@[[GLOBAL_TYPE_SCOPE]]::@[[TYPE1]], !hw.struct<b: i1>>>>) {
  firrtl.module @Foo(in %in: !firrtl.uint<1>, out %sink: !firrtl.bundle<a: bundle<b: uint<1>>>) {
    // CHECK-NEXT:  sv.wire  : !hw.inout<typealias<@[[GLOBAL_TYPE_SCOPE]]::@[[TYPE2]], !hw.struct<a: !hw.typealias<@[[GLOBAL_TYPE_SCOPE]]::@[[TYPE1]], !hw.struct<b: i1>>>>>
    // CHECK-NEXT:  sv.read_inout {{.+}} : !hw.inout<typealias<@[[GLOBAL_TYPE_SCOPE]]::@[[TYPE2]], !hw.struct<a: !hw.typealias<@[[GLOBAL_TYPE_SCOPE]]::@[[TYPE1]], !hw.struct<b: i1>>>>>
    // CHECK-NEXT:  sv.struct_field_inout {{.+}}["a"] : !hw.inout<typealias<@[[GLOBAL_TYPE_SCOPE]]::@[[TYPE2]], !hw.struct<a: !hw.typealias<@[[GLOBAL_TYPE_SCOPE]]::@[[TYPE1]], !hw.struct<b: i1>>>>>
    // CHECK-NEXT:  sv.struct_field_inout {{.+}}["b"] : !hw.inout<typealias<@[[GLOBAL_TYPE_SCOPE]]::@[[TYPE1]], !hw.struct<b: i1>>>
    %0 = firrtl.subfield %sink(0) : (!firrtl.bundle<a: bundle<b: uint<1>>>) -> !firrtl.bundle<b: uint<1>>
    %1 = firrtl.subfield %0(0) : (!firrtl.bundle<b: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %1, %in : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK-LABEL:   hw.module @UseFoo() ->
  // CHECK-SAME:      (sink: !hw.typealias<@[[GLOBAL_TYPE_SCOPE]]::@[[TYPE2]], !hw.struct<a: !hw.typealias<@[[GLOBAL_TYPE_SCOPE]]::@[[TYPE1]], !hw.struct<b: i1>>>>) {
  firrtl.module @UseFoo(out %sink: !firrtl.bundle<a: bundle<b: uint<1>>>) {
    // CHECK:           hw.instance "fetch" @Foo
    // CHECK-SAME:         -> (sink: !hw.typealias<@[[GLOBAL_TYPE_SCOPE]]::@[[TYPE2]], !hw.struct<a: !hw.typealias<@[[GLOBAL_TYPE_SCOPE]]::@[[TYPE1]], !hw.struct<b: i1>>>>)
    %a, %b = firrtl.instance fetch @Foo(in in: !firrtl.uint<1>, out sink: !firrtl.bundle<a: bundle<b: uint<1>>>)
  }

  // CHECK-LABEL:   hw.type_scope
  // CHECK:         @[[MODULE_TYPE_SCOPE_BAR:.+]] {
  // CHECK-NEXT:      hw.typedecl @[[TYPE1:.+]] : !hw.struct<a: i1>
  // CHECK-NEXT:    }

  // CHECK-LABEL:   hw.module @Bar() {
  firrtl.module @Bar() {
    // CHECK-NEXT:      sv.wire  : !hw.inout<typealias<@[[MODULE_TYPE_SCOPE_BAR]]::@[[TYPE1]], !hw.struct<a: i1>>>
    %a = firrtl.wire : !firrtl.bundle<a: uint<1>>
  }

  // CHECK: hw.module.extern @Ext(%inA: !hw.typealias<@[[GLOBAL_TYPE_SCOPE]]::@[[TYPE3]], !hw.struct<ext: i1>>)
  firrtl.extmodule @Ext(in inA: !firrtl.bundle<ext: uint<1>>)
}