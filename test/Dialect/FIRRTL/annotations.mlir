// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-lower-annotations)' -split-input-file %s |FileCheck %s

// circt.test copies the annotation to the target
// circt.testNT puts the targetless annotation on the circuit

firrtl.circuit "Aggregates" attributes {annotations = [
  {class = "circt.test", target = "~Aggregates|Aggregates>vector[1][1][1]"},
  {class = "circt.test", target = "~Aggregates|Aggregates>bundle.a.b.c"}
  ]} {
  firrtl.module @Aggregates() {
    // CHECK: {annotations = [{circt.fieldID = 14 : i32, class = "circt.test"}]}
    %vector = firrtl.wire  : !firrtl.vector<vector<vector<uint<1>, 2>, 2>, 2>
    // CHECK: {annotations = [{circt.fieldID = 3 : i32, class = "circt.test"}]}
    %bundle = firrtl.wire  : !firrtl.bundle<a: bundle<b: bundle<c: uint<1>>>>
  }
}
  
// -----

// A non-local annotation should work.

// CHECK-LABEL: firrtl.circuit "FooNL"
// CHECK: firrtl.nla @nla_1 [#hw.innerNameRef<@FooNL::@baz>, #hw.innerNameRef<@BazNL::@bar>, @BarNL]
// CHECK: firrtl.nla @nla_0 [#hw.innerNameRef<@FooNL::@baz>, #hw.innerNameRef<@BazNL::@bar>, #hw.innerNameRef<@BarNL::@w>]
// CHECK: firrtl.nla @nla [#hw.innerNameRef<@FooNL::@baz>, #hw.innerNameRef<@BazNL::@bar>, #hw.innerNameRef<@BarNL::@w2>]
// CHECK: firrtl.module @BarNL
// CHECK: %w = firrtl.wire sym @w {annotations = [{circt.nonlocal = @nla_0, class = "circt.test", nl = "nl"}]}
// CHECK: %w2 = firrtl.wire sym @w2 {annotations = [{circt.fieldID = 5 : i32, circt.nonlocal = @nla, class = "circt.test", nl = "nl2"}]} : !firrtl.bundle<a: uint, b: vector<uint, 4>>
// CHECK: firrtl.instance bar sym @bar {annotations = [{circt.nonlocal = @nla, class = "circt.nonlocal"}, {circt.nonlocal = @nla_0, class = "circt.nonlocal"}, {circt.nonlocal = @nla_1, class = "circt.nonlocal"}]} @BarNL()
// CHECK: firrtl.instance baz sym @baz {annotations = [{circt.nonlocal = @nla, class = "circt.nonlocal"}, {circt.nonlocal = @nla_0, class = "circt.nonlocal"}, {circt.nonlocal = @nla_1, class = "circt.nonlocal"}]} @BazNL()
// CHECK: firrtl.module @FooL
// CHECK: %w3 = firrtl.wire {annotations = [{class = "circt.test", nl = "nl3"}]}
firrtl.circuit "FooNL"  attributes {annotations = [
  {class = "circt.test", nl = "nl", target = "~FooNL|FooNL/baz:BazNL/bar:BarNL"},
  {class = "circt.test", nl = "nl", target = "~FooNL|FooNL/baz:BazNL/bar:BarNL>w"},
  {class = "circt.test", nl = "nl2", target = "~FooNL|FooNL/baz:BazNL/bar:BarNL>w2.b[2]"},
  {class = "circt.test", nl = "nl3", target = "~FooNL|FooL>w3"}
  ]}  {
  firrtl.module @BarNL() {
    %w = firrtl.wire  sym @w : !firrtl.uint
    %w2 = firrtl.wire sym @w2 : !firrtl.bundle<a: uint, b: vector<uint, 4>>
    firrtl.skip
  }
  firrtl.module @BazNL() {
    firrtl.instance bar sym @bar @BarNL()
  }
  firrtl.module @FooNL() {
    firrtl.instance baz sym @baz @BazNL()
  }
  firrtl.module @FooL() {
    %w3 = firrtl.wire: !firrtl.uint
  }
}

// -----

// Non-local annotations on memory ports should work.

// CHECK-LABEL: firrtl.circuit "MemPortsNL"
// CHECK: firrtl.nla @nla [#hw.innerNameRef<@MemPortsNL::@child>, #hw.innerNameRef<@Child::@bar>]
// CHECK: firrtl.module @Child()
// CHECK:   %bar_r = firrtl.mem sym @bar
// CHECK-SAME: portAnnotations = {{\[}}[{circt.nonlocal = @nla, class = "circt.test", nl = "nl"}]]
// CHECK: firrtl.module @MemPortsNL()
// CHECK:   firrtl.instance child sym @child
// CHECK-SAME: annotations = [{circt.nonlocal = @nla, class = "circt.nonlocal"}]
firrtl.circuit "MemPortsNL" attributes {annotations = [
  {class = "circt.test", nl = "nl", target = "~MemPortsNL|MemPortsNL/child:Child>bar.r"}
  ]}  {
  firrtl.module @Child() {
    %bar_r = firrtl.mem Undefined  {depth = 16 : i64, name = "bar", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
  }
  firrtl.module @MemPortsNL() {
    firrtl.instance child @Child()
  }
}

// -----

// Annotations on ports should work.
firrtl.circuit "Test" attributes {annotations = [
  {class = "circt.test", target = "~Test|PortTest>in"}
  ]} {
  firrtl.module @PortTest(in %in : !firrtl.uint<1>) {}
  firrtl.module @Test() {
    %portttest_in = firrtl.instance porttest @PortTest(in in : !firrtl.uint<1>)
  }
}

// -----

// Subannotations on ports should work.
firrtl.circuit "Test" attributes {annotations = [
  {class = "circt.test", target = "~Test|PortTest>in.a"}
  ]} {
  // CHECK: firrtl.module @PortTest(in %in: !firrtl.bundle<a: uint<1>> [{circt.fieldID = 1 : i32, class = "circt.test"}])
  firrtl.module @PortTest(in %in : !firrtl.bundle<a: uint<1>>) {}
  firrtl.module @Test() {
    %portttest_in = firrtl.instance porttest @PortTest(in in : !firrtl.bundle<a: uint<1>>)
  }
}
// -----

// Annotations on instances should be moved to the target module.
firrtl.circuit "Test" attributes {annotations = [
  {class = "circt.test", target = "~Test|Test>exttest"}
  ]} {
  // CHECK: firrtl.nla @nla [#hw.innerNameRef<@Test::@exttest>, @ExtTest]
  // CHECK: firrtl.extmodule @ExtTest() attributes {annotations = [{circt.nonlocal = @nla, class = "circt.test"}]}
  firrtl.extmodule @ExtTest()
  
  firrtl.module @Test() {
    // CHECK: firrtl.instance exttest sym @exttest  {annotations = [{circt.nonlocal = @nla, class = "circt.nonlocal"}]} @ExtTest()
    firrtl.instance exttest @ExtTest()
  }
}

// -----

// Annotations on instances should be moved to the target module.
firrtl.circuit "Test" attributes {annotations = [
  {class = "circt.test", target = "~Test|Test>exttest.in"}
  ]} {
  // CHECK: firrtl.nla @nla [#hw.innerNameRef<@Test::@exttest>, #hw.innerNameRef<@ExtTest::@in>]
  // CHECK: firrtl.extmodule @ExtTest(in in: !firrtl.uint<1> sym @in [{circt.nonlocal = @nla, class = "circt.test"}])
  firrtl.extmodule @ExtTest(in in: !firrtl.uint<1>)

  firrtl.module @Test() {
    // CHECK: %exttest_in = firrtl.instance exttest sym @exttest  {annotations = [{circt.nonlocal = @nla, class = "circt.nonlocal"}]} @ExtTest(in in: !firrtl.uint<1>)
    firrtl.instance exttest @ExtTest(in in : !firrtl.uint<1>)
  }
}
