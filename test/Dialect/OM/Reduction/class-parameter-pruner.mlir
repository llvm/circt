// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "om.object @Foo" --include om-class-parameter-pruner --keep-best=0 | FileCheck %s

// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129

// This test verifies that the OMClassParameterPruner can remove unused parameters
// from om.class definitions and update all instantiations accordingly.

module {
  // CHECK-LABEL: om.class @Foo(%used: !om.integer) -> (out: !om.integer)
  om.class @Foo(%unused1: !om.string, %used: !om.integer, %unused2: i1) -> (out: !om.integer) {
    // Only %used is actually used in the class body
    // CHECK: om.class.fields %used
    om.class.fields %used : !om.integer
  }

  // The unused %basepath parameter should be removed from @Bar
  // CHECK-LABEL: om.class @Bar()
  // CHECK-NOT: %basepath
  om.class @Bar(%basepath: !om.frozenbasepath) -> (result: !om.class.type<@Foo>) {
    %0 = om.constant "unused_string" : !om.string
    %1 = om.constant #om.integer<42 : si64> : !om.integer
    %2 = om.constant true

    // CHECK: om.object @Foo
    %3 = om.object @Foo(%0, %1, %2) : (!om.string, !om.integer, i1) -> !om.class.type<@Foo>

    om.class.fields %3 : !om.class.type<@Foo>
  }
}
