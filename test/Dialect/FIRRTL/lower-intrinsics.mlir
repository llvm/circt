// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-intrinsics))' %s   | FileCheck %s

// CHECK-LABEL: "Foo"
firrtl.circuit "Foo" {
  // CHECK-NOT: NameDoesNotMatter
  firrtl.extmodule @NameDoesNotMatter(in i : !firrtl.clock, out size : !firrtl.uint<32>) attributes
                                     {annotations = [{class = "circt.intrinsic", intrinsic = "circt.sizeof"}]}
  // CHECK-NOT: NameDoesNotMatter2
  firrtl.extmodule @NameDoesNotMatter2(in i : !firrtl.clock, out found : !firrtl.uint<1>) attributes
                                     {annotations = [{class = "circt.intrinsic", intrinsic = "circt.isX"}]}

  // CHECK: Foo
  firrtl.module @Foo(in %clk : !firrtl.clock, out %o1 : !firrtl.uint<32>, out %o2 : !firrtl.uint<1>) {
    %i1, %size = firrtl.instance "" @NameDoesNotMatter(in i : !firrtl.clock, out size : !firrtl.uint<32>)
    // CHECK-NOT: NameDoesNotMatter
    // CHECK: firrtl.int.sizeof
    firrtl.strictconnect %i1, %clk : !firrtl.clock
    firrtl.strictconnect %o1, %size : !firrtl.uint<32>

    %i2, %found = firrtl.instance "" @NameDoesNotMatter2(in i : !firrtl.clock, out found : !firrtl.uint<1>)
    // CHECK-NOT: NameDoesNotMatter2
    // CHECK: firrtl.int.isX
    firrtl.strictconnect %i2, %clk : !firrtl.clock
    firrtl.strictconnect %o2, %found : !firrtl.uint<1>
  }
}
