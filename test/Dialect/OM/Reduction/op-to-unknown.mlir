// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "add" --include om-op-to-unknown --keep-best=0 | FileCheck %s

// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129

// This test verifies that the generic OMOpToUnknown pattern replaces ANY
// single-result OM operation with om.unknown, operating at the lowest level
// on Operation* without needing concrete op types.

module {
  // CHECK-LABEL: om.class @TestClass(%input: !om.integer) -> (result: !om.integer) {
  om.class @TestClass(%input: !om.integer) -> (result: !om.integer) {
    // CHECK-NEXT: %[[UNKNOWN_0:.+]] = om.unknown : !om.integer
    // CHECK-NEXT: %[[UNKNOWN_1:.+]] = om.unknown : !om.integer
    %const_42 = om.constant #om.integer<42 : si64> : !om.integer
    %const_43 = om.constant #om.integer<43 : si64> : !om.integer

    // Test constant replacement
    // CHECK: %[[ADD:.+]] = om.integer.add %[[UNKNOWN_0]], %[[UNKNOWN_1]] : !om.integer
    %add = om.integer.add %const_42, %const_43 : !om.integer

    om.class.fields %add : !om.integer
  }
}
