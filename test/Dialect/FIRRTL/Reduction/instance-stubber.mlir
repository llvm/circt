// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg true --include instance-stubber | FileCheck %s

firrtl.circuit "Foo" {
  firrtl.module @Bar(
    out %a: !firrtl.probe<uint<1>>,
    out %b: !firrtl.rwprobe<uint<2>>,
    out %str_out: !firrtl.string,
    out %anyref_out: !firrtl.anyref
  ) {}
  firrtl.module @Foo() {
    %bar_a, %bar_b, %bar_str, %bar_anyref = firrtl.instance bar @Bar(
      out a: !firrtl.probe<uint<1>>,
      out b: !firrtl.rwprobe<uint<2>>,
      out str_out: !firrtl.string,
      out anyref_out: !firrtl.anyref
    )
  }
}

// CHECK-LABEL: firrtl.module @Foo() {

// Check that probe types get proper wire infrastructure with invalidation.
// CHECK: %bar_a = firrtl.wire : !firrtl.probe<uint<1>>
// CHECK: %[[UNDERLYING_A:.+]] = firrtl.wire : !firrtl.uint<1>
// CHECK: %[[REF_A:.+]] = firrtl.ref.send %[[UNDERLYING_A]] : !firrtl.uint<1>
// CHECK: firrtl.ref.define %bar_a, %[[REF_A]] : !firrtl.probe<uint<1>>
// CHECK: %[[INVALID_UI1:.+]] = firrtl.invalidvalue : !firrtl.uint<1>
// CHECK: firrtl.matchingconnect %[[UNDERLYING_A]], %[[INVALID_UI1]] : !firrtl.uint<1>

// Check that rwprobe types get proper forceable wire infrastructure with
// invalidation.
// CHECK: %bar_b = firrtl.wire : !firrtl.rwprobe<uint<2>>
// CHECK: %[[UNDERLYING_B:.+]]:2 = firrtl.wire forceable : !firrtl.uint<2>, !firrtl.rwprobe<uint<2>>
// CHECK: firrtl.ref.define %bar_b, %[[UNDERLYING_B]]#1 : !firrtl.rwprobe<uint<2>>
// CHECK: %[[INVALID_UI2:.+]] = firrtl.invalidvalue : !firrtl.uint<2>
// CHECK: firrtl.matchingconnect %[[UNDERLYING_B]]#0, %[[INVALID_UI2]] : !firrtl.uint<2>

// Check that property types get wires with UnknownValueOp tie-offs
// CHECK: %bar_str_out = firrtl.wire : !firrtl.string
// CHECK: %[[UNKNOWN_STR:.+]] = firrtl.unknown : !firrtl.string
// CHECK: firrtl.propassign %bar_str_out, %[[UNKNOWN_STR]] : !firrtl.string
// CHECK: %bar_anyref_out = firrtl.wire : !firrtl.anyref
// CHECK: %[[UNKNOWN_ANYREF:.+]] = firrtl.unknown : !firrtl.anyref
// CHECK: firrtl.propassign %bar_anyref_out, %[[UNKNOWN_ANYREF]] : !firrtl.anyref
