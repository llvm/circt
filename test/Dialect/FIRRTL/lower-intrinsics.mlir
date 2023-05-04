// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-intrinsics))' %s   | FileCheck %s

// CHECK-LABEL: "Foo"
firrtl.circuit "Foo" {
  // CHECK-NOT: NameDoesNotMatter
  firrtl.extmodule @NameDoesNotMatter(in i : !firrtl.clock, out size : !firrtl.uint<32>) attributes
                                     {annotations = [{class = "circt.Intrinsic", intrinsic = "circt.sizeof"}]}
  // CHECK-NOT: NameDoesNotMatter2
  firrtl.extmodule @NameDoesNotMatter2(in i : !firrtl.clock, out found : !firrtl.uint<1>) attributes
                                     {annotations = [{class = "circt.Intrinsic", intrinsic = "circt.isX"}]}
  // CHECK-NOT: NameDoesNotMatter3
  firrtl.extmodule @NameDoesNotMatter3<FORMAT: none = "foo">(out found : !firrtl.uint<1>) attributes
                                     {annotations = [{class = "circt.Intrinsic", intrinsic = "circt.plusargs.test"}]}
  // CHECK-NOT: NameDoesNotMatter4
  firrtl.extmodule @NameDoesNotMatter4<FORMAT: none = "foo">(out found : !firrtl.uint<1>, out result: !firrtl.uint<5>) attributes
                                     {annotations = [{class = "circt.Intrinsic", intrinsic = "circt.plusargs.value"}]}

  // CHECK: Foo
  firrtl.module @Foo(in %clk : !firrtl.clock, out %s : !firrtl.uint<32>, out %io1 : !firrtl.uint<1>, out %io2 : !firrtl.uint<1>, out %io3 : !firrtl.uint<1>, out %io4 : !firrtl.uint<5>) {
    %i1, %size = firrtl.instance "" @NameDoesNotMatter(in i : !firrtl.clock, out size : !firrtl.uint<32>)
    // CHECK-NOT: NameDoesNotMatter
    // CHECK: firrtl.int.sizeof
    firrtl.strictconnect %i1, %clk : !firrtl.clock
    firrtl.strictconnect %s, %size : !firrtl.uint<32>

    %i2, %found2 = firrtl.instance "" @NameDoesNotMatter2(in i : !firrtl.clock, out found : !firrtl.uint<1>)
    // CHECK-NOT: NameDoesNotMatter2
    // CHECK: firrtl.int.isX
    firrtl.strictconnect %i2, %clk : !firrtl.clock
    firrtl.strictconnect %io1, %found2 : !firrtl.uint<1>

    %found3 = firrtl.instance "" @NameDoesNotMatter3(out found : !firrtl.uint<1>)
    // CHECK-NOT: NameDoesNotMatter3
    // CHECK: firrtl.int.plusargs.test "foo"
    firrtl.strictconnect %io2, %found3 : !firrtl.uint<1>

    %found4, %result1 = firrtl.instance "" @NameDoesNotMatter4(out found : !firrtl.uint<1>, out result: !firrtl.uint<5>)
    // CHECK-NOT: NameDoesNotMatter4
    // CHECK: firrtl.int.plusargs.value "foo" : !firrtl.uint<5>
    firrtl.strictconnect %io3, %found4 : !firrtl.uint<1>
    firrtl.strictconnect %io4, %result1 : !firrtl.uint<5>
  }

  // CHECK-NOT: NameDoesNotMatte5
  firrtl.intmodule @NameDoesNotMatter5(in i : !firrtl.clock, out size : !firrtl.uint<32>) attributes
                                     {intrinsic = "circt.sizeof"}
  // CHECK-NOT: NameDoesNotMatter6
  firrtl.intmodule @NameDoesNotMatter6(in i : !firrtl.clock, out found : !firrtl.uint<1>) attributes
                                     {intrinsic = "circt.isX"}
  // CHECK-NOT: NameDoesNotMatter7
  firrtl.intmodule @NameDoesNotMatter7<FORMAT: none = "foo">(out found : !firrtl.uint<1>) attributes
                                     {intrinsic = "circt.plusargs.test"}
  // CHECK-NOT: NameDoesNotMatter8
  firrtl.intmodule @NameDoesNotMatter8<FORMAT: none = "foo">(out found : !firrtl.uint<1>, out result: !firrtl.uint<5>) attributes
                                     {intrinsic = "circt.plusargs.value"}

  // CHECK: Bar
  firrtl.module @Bar(in %clk : !firrtl.clock, out %s : !firrtl.uint<32>, out %io1 : !firrtl.uint<1>, out %io2 : !firrtl.uint<1>, out %io3 : !firrtl.uint<1>, out %io4 : !firrtl.uint<5>) {
    %i1, %size = firrtl.instance "" @NameDoesNotMatter5(in i : !firrtl.clock, out size : !firrtl.uint<32>)
    // CHECK-NOT: NameDoesNotMatter5
    // CHECK: firrtl.int.sizeof
    firrtl.strictconnect %i1, %clk : !firrtl.clock
    firrtl.strictconnect %s, %size : !firrtl.uint<32>

    %i2, %found2 = firrtl.instance "" @NameDoesNotMatter6(in i : !firrtl.clock, out found : !firrtl.uint<1>)
    // CHECK-NOT: NameDoesNotMatter6
    // CHECK: firrtl.int.isX
    firrtl.strictconnect %i2, %clk : !firrtl.clock
    firrtl.strictconnect %io1, %found2 : !firrtl.uint<1>

    %found3 = firrtl.instance "" @NameDoesNotMatter7(out found : !firrtl.uint<1>)
    // CHECK-NOT: NameDoesNotMatter7
    // CHECK: firrtl.int.plusargs.test "foo"
    firrtl.strictconnect %io2, %found3 : !firrtl.uint<1>

    %found4, %result1 = firrtl.instance "" @NameDoesNotMatter8(out found : !firrtl.uint<1>, out result: !firrtl.uint<5>)
    // CHECK-NOT: NameDoesNotMatter8
    // CHECK: firrtl.int.plusargs.value "foo" : !firrtl.uint<5>
    firrtl.strictconnect %io3, %found4 : !firrtl.uint<1>
    firrtl.strictconnect %io4, %result1 : !firrtl.uint<5>
  }

  // CHECK-NOT: ClockGate0
  // CHECK-NOT: ClockGate1
  firrtl.extmodule @ClockGate0(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {annotations = [{class = "circt.Intrinsic", intrinsic = "circt.clock_gate"}]}
  firrtl.intmodule @ClockGate1(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {intrinsic = "circt.clock_gate"}

  // CHECK: ClockGate
  firrtl.module @ClockGate(in %clk: !firrtl.clock, in %en: !firrtl.uint<1>) {
    // CHECK-NOT: ClockGate0
    // CHECK: firrtl.int.clock_gate
    %in1, %en1, %out1 = firrtl.instance "" @ClockGate0(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    firrtl.strictconnect %in1, %clk : !firrtl.clock
    firrtl.strictconnect %en1, %en : !firrtl.uint<1>

    // CHECK-NOT: ClockGate1
    // CHECK: firrtl.int.clock_gate
    %in2, %en2, %out2 = firrtl.instance "" @ClockGate1(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    firrtl.strictconnect %in2, %clk : !firrtl.clock
    firrtl.strictconnect %en2, %en : !firrtl.uint<1>
  }
}
