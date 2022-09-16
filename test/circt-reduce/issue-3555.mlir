// RUN: circt-reduce %s --test %S/test.sh --test-arg firtool --test-arg "error: sink \"x1.x\" not fully initialized" --keep-best=0 --include root-port-pruner --test-must-fail | FileCheck %s

// https://github.com/llvm/circt/issues/3555
firrtl.circuit "Foo"  {
  // CHECK-LABEL: firrtl.module @Foo
  // CHECK-SAME:  () {
  firrtl.module @Foo(in %x: !firrtl.uint<1>, out %y: !firrtl.uint<1>) {
    %x1_x = firrtl.wire   : !firrtl.uint<1>
    %invalid_ui1 = firrtl.invalidvalue : !firrtl.uint<1>
    // CHECK-NOT: firrtl.strictconnect %y
    firrtl.strictconnect %y, %invalid_ui1 : !firrtl.uint<1>
  }
  // CHECK: }
}
