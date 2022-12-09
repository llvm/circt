// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-intrinsics))' %s   | FileCheck %s

// CHECK-LABEL: "Foo"
firrtl.circuit "Foo" {
  // CHECK-NOT: NameDoesNotMatter
  firrtl.extmodule @NameDoesNotMatter(in i : !firrtl.clock, out size : !firrtl.uint<32>) attributes
                                     {annotations = [{class = "circt.intrinsic", intrinsic = "circt.sizeof"}]}
  // CHECK-NOT: NameDoesNotMatter2
  firrtl.extmodule @NameDoesNotMatter2(in i : !firrtl.clock, out found : !firrtl.uint<1>) attributes
                                     {annotations = [{class = "circt.intrinsic", intrinsic = "circt.isX"}]}
  // CHECK-NOT: NameDoesNotMatter3
  firrtl.extmodule @NameDoesNotMatter3<FORMAT: none = "foo">(out found : !firrtl.uint<1>) attributes
                                     {annotations = [{class = "circt.intrinsic", intrinsic = "circt.plusargs.test"}]}
  // CHECK-NOT: NameDoesNotMatter4
  firrtl.extmodule @NameDoesNotMatter4<FORMAT: none = "foo">(out found : !firrtl.uint<1>, out result: !firrtl.uint<5>) attributes
                                     {annotations = [{class = "circt.intrinsic", intrinsic = "circt.plusargs.value"}]}

  // CHECK: Foo
  firrtl.module @Foo(in %clk : !firrtl.clock) {
    %i1, %size = firrtl.instance "" @NameDoesNotMatter(in i : !firrtl.clock, out size : !firrtl.uint<32>)
    // CHECK-NOT: NameDoesNotMatter
    // CHECK: firrtl.int.sizeof
    firrtl.strictconnect %i1, %clk : !firrtl.clock

    %i2, %found2 = firrtl.instance "" @NameDoesNotMatter2(in i : !firrtl.clock, out found : !firrtl.uint<1>)
    // CHECK-NOT: NameDoesNotMatter2
    // CHECK: firrtl.int.isX
    firrtl.strictconnect %i2, %clk : !firrtl.clock

    %found3 = firrtl.instance "" @NameDoesNotMatter3(out found : !firrtl.uint<1>)
    // CHECK-NOT: NameDoesNotMatter3
    // CHECK: firrtl.int.plusargs.test "foo"

    %found4, %result1 = firrtl.instance "" @NameDoesNotMatter4(out found : !firrtl.uint<1>, out result: !firrtl.uint<5>)
    // CHECK-NOT: NameDoesNotMatter4
    // CHECK: %5:2 = firrtl.int.plusargs.value "foo" : !firrtl.uint<5>
  }
}
