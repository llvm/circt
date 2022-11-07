// RUN: circt-opt -mlir-print-debuginfo -mlir-print-local-scope -pass-pipeline='builtin.module(firrtl.circuit(firrtl-dedup))' %s | FileCheck %s

firrtl.circuit "Test" {
// CHECK-LABEL: @Dedup0()
firrtl.module @Dedup0() {
  // CHECK: %w = firrtl.wire  : !firrtl.uint<1> loc(fused["foo", "bar"])
  %w = firrtl.wire : !firrtl.uint<1> loc("foo")
} loc("dedup0")
// CHECK: loc(fused["dedup0", "dedup1"])
// CHECK-NOT: @Dedup1()
firrtl.module @Dedup1() {
  %w = firrtl.wire : !firrtl.uint<1> loc("bar")
} loc("dedup1")
firrtl.module @Test() {
  firrtl.instance dedup0 @Dedup0()
  firrtl.instance dedup1 @Dedup1()
}
}
