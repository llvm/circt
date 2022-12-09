// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-intrinsics))' %s   | FileCheck %s

// CHECK-LABEL: "Foo"
firrtl.circuit "Foo" {
  // CHECK-NOT: NameDoesNotMatter
  firrtl.extmodule @NameDoesNotMatter(in i : !firrtl.clock, out size : !firrtl.uint<32>) attributes
                                     {annotations = [{class = "circt.intrinsic", intrinsic = "circt.sizeof"}]}

  // CHECK: Foo
  firrtl.module @Foo(in %clk : !firrtl.clock, out %o : !firrtl.uint<32>) {
    %i, %size = firrtl.instance "" @NameDoesNotMatter(in i : !firrtl.clock, out size : !firrtl.uint<32>)
    // CHECK-NOT: instance
    // CHECK: firrtl.sizeof
    firrtl.strictconnect %i, %clk : !firrtl.clock
    firrtl.strictconnect %o, %size : !firrtl.uint<32>
  }
}
