// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "om.any_cast.*Foo" --include om-object-to-unknown --keep-best=0 | FileCheck %s

// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129

// This test verifies that the OMObjectToUnknownReplacer can replace om.object
// instantiations with om.unknown when the object is only used in om.any_cast
// operations or not used at all.

module {
  // CHECK-LABEL: om.class @Foo
  om.class @Foo(%input: !om.integer) -> (output: !om.integer) {
    om.class.fields %input : !om.integer
  }

  // CHECK-LABEL: om.class @Bar
  om.class @Bar(%basepath: !om.frozenbasepath) -> (result: !om.any) {
    %0 = om.constant #om.integer<42 : si64> : !om.integer

    // This object is only used in an any_cast, so it should be replaced with om.unknown
    // CHECK-NOT: om.object @Foo
    // CHECK: %[[UNKNOWN:.+]] = om.unknown : !om.class.type<@Foo>
    %1 = om.object @Foo(%0) : (!om.integer) -> !om.class.type<@Foo>

    // CHECK: om.any_cast %[[UNKNOWN]]
    %2 = om.any_cast %1 : (!om.class.type<@Foo>) -> !om.any

    om.class.fields %2 : !om.any
  }

  // CHECK-LABEL: om.class @Baz
  om.class @Baz(%basepath: !om.frozenbasepath) -> (result: !om.any) {
    %0 = om.constant #om.integer<99 : si64> : !om.integer

    // This object is not used at all. It should _still_ be replaced.
    // CHECK-NOT: om.object @Foo
    %1 = om.object @Foo(%0) : (!om.integer) -> !om.class.type<@Foo>

    %2 = om.constant "dummy" : !om.string
    %3 = om.any_cast %2 : (!om.string) -> !om.any

    om.class.fields %3 : !om.any
  }
}
