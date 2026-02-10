// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "om.class @SimplifyTest" --include om-anycast-of-unknown-simplifier --keep-best=0 | FileCheck %s

// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129

// This test verifies that the OMAnyCastOfUnknownSimplifier can simplify chains
// of om.unknown -> om.any_cast by replacing them with a direct om.unknown of
// the target type.

module {
  // CHECK-LABEL: om.class @Foo
  om.class @Foo() {
    om.class.fields
  }

  // CHECK-LABEL: om.class @SimplifyTest
  om.class @SimplifyTest(%basepath: !om.frozenbasepath) -> (result1: !om.any, result2: !om.any) {
    // This pattern should be simplified - the any_cast is replaced with om.unknown : !om.any
    // The original om.unknown becomes dead code (removed by generic dead code elimination)
    // CHECK: %[[UNKNOWN1:.+]] = om.unknown : !om.any
    %0 = om.unknown : !om.class.type<@Foo>
    // CHECK-NOT: om.any_cast
    %1 = om.any_cast %0 : (!om.class.type<@Foo>) -> !om.any

    // Another pattern to simplify
    // CHECK: %[[UNKNOWN2:.+]] = om.unknown : !om.any
    %2 = om.unknown : !om.string
    %3 = om.any_cast %2 : (!om.string) -> !om.any

    // CHECK: om.class.fields %[[UNKNOWN1]], %[[UNKNOWN2]]
    om.class.fields %1, %3 : !om.any, !om.any
  }

  // CHECK-LABEL: om.class @NoSimplifyTest
  om.class @NoSimplifyTest(%basepath: !om.frozenbasepath) -> (result: !om.any) {
    %0 = om.constant "not_unknown" : !om.string

    // This should NOT be simplified because the input is not om.unknown
    // CHECK: om.any_cast
    %1 = om.any_cast %0 : (!om.string) -> !om.any

    om.class.fields %1 : !om.any
  }
}
