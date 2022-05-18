// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-lower-annotations)' -split-input-file %s |FileCheck %s

// circt.test copies the annotation to the target
// circt.testNT puts the targetless annotation on the circuit

firrtl.circuit "Aggregates" attributes {rawAnnotations = [
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
firrtl.circuit "FooNL"  attributes {rawAnnotations = [
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
firrtl.circuit "MemPortsNL" attributes {rawAnnotations = [
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
firrtl.circuit "Test" attributes {rawAnnotations = [
  {class = "circt.test", target = "~Test|PortTest>in"}
  ]} {
  firrtl.module @PortTest(in %in : !firrtl.uint<1>) {}
  firrtl.module @Test() {
    %portttest_in = firrtl.instance porttest @PortTest(in in : !firrtl.uint<1>)
  }
}

// -----

// Subannotations on ports should work.
firrtl.circuit "Test" attributes {rawAnnotations = [
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
firrtl.circuit "Test" attributes {rawAnnotations = [
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
firrtl.circuit "Test" attributes {rawAnnotations = [
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

// -----

// DontTouchAnnotations create symbols on the things they target.

firrtl.circuit "Foo"  attributes {
  rawAnnotations = [
    {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_0"},
    {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_1"},
    {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_2"},
    {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_3"},
    {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_4"},
    {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_5"},
    {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_6"},
    {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_8"},
    {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_9.a"}]} {
  // CHECK: firrtl.module @Foo
  firrtl.module @Foo(in %reset: !firrtl.uint<1>, in %clock: !firrtl.clock) {
    // CHECK-NEXT: %_T_0 = firrtl.wire sym @_T_0
    %_T_0 = firrtl.wire  : !firrtl.uint<1>
    // CHECK-NEXT: %_T_1 = firrtl.node sym @_T_1
    %_T_1 = firrtl.node %_T_0  : !firrtl.uint<1>
    // CHECK-NEXT: %_T_2 = firrtl.reg sym @_T_2
    %_T_2 = firrtl.reg %clock  : !firrtl.uint<1>
    %c0_ui4 = firrtl.constant 0 : !firrtl.uint<4>
    // CHECK: %_T_3 = firrtl.regreset sym @_T_3
    %_T_3 = firrtl.regreset %clock, %reset, %c0_ui4  : !firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>
    // CHECK-NEXT: %_T_4 = chirrtl.seqmem sym @_T_4
    %_T_4 = chirrtl.seqmem Undefined  : !chirrtl.cmemory<vector<uint<1>, 9>, 256>
    // CHECK-NEXT: %_T_5 = chirrtl.combmem sym @_T_5
    %_T_5 = chirrtl.combmem  : !chirrtl.cmemory<vector<uint<1>, 9>, 256>
    // CHECK: chirrtl.memoryport Infer %_T_5 {annotations =
    // CHECK-SAME: {class = "firrtl.transforms.DontTouchAnnotation"}
    %_T_6_data, %_T_6_port = chirrtl.memoryport Infer %_T_5  {name = "_T_6"} : (!chirrtl.cmemory<vector<uint<1>, 9>, 256>) -> (!firrtl.vector<uint<1>, 9>, !chirrtl.cmemoryport)
    chirrtl.memoryport.access %_T_6_port[%reset], %clock : !chirrtl.cmemoryport, !firrtl.uint<1>, !firrtl.clock
    // CHECK: firrtl.mem sym @_T_8
    %_T_8_w = firrtl.mem Undefined  {depth = 8 : i64, name = "_T_8", portNames = ["w"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<4>, mask: uint<1>>
    %aggregate = firrtl.wire  : !firrtl.bundle<a: uint<1>>
    // CHECK: %_T_9 = firrtl.node %aggregate {annotations = [{circt.fieldID = 1 : i32, class = "firrtl.transforms.DontTouchAnnotation"}]}
    %_T_9 = firrtl.node %aggregate  : !firrtl.bundle<a: uint<1>>
  }
}

// -----

firrtl.circuit "GCTInterface"  attributes {annotations = [{unrelatedAnnotation}], rawAnnotations = [{class = "sifive.enterprise.grandcentral.GrandCentralView$SerializedViewAnnotation", companion = "~GCTInterface|view_companion", name = "view", parent = "~GCTInterface|GCTInterface", view = {class = "sifive.enterprise.grandcentral.AugmentedBundleType", defName = "ViewName", elements = [{description = "the register in GCTInterface", name = "register", tpe = {class = "sifive.enterprise.grandcentral.AugmentedBundleType", defName = "Register", elements = [{name = "_2", tpe = {class = "sifive.enterprise.grandcentral.AugmentedVectorType", elements = [{class = "sifive.enterprise.grandcentral.AugmentedGroundType", ref = {circuit = "GCTInterface", component = [{class = "firrtl.annotations.TargetToken$Field", value = "_2"}, {class = "firrtl.annotations.TargetToken$Index", value = 0 : i64}], module = "GCTInterface", path = [], ref = "r"}, tpe = {class = "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"}}, {class = "sifive.enterprise.grandcentral.AugmentedGroundType", ref = {circuit = "GCTInterface", component = [{class = "firrtl.annotations.TargetToken$Field", value = "_2"}, {class = "firrtl.annotations.TargetToken$Index", value = 1 : i64}], module = "GCTInterface", path = [], ref = "r"}, tpe = {class = "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"}}]}}, {name = "_0_inst", tpe = {class = "sifive.enterprise.grandcentral.AugmentedBundleType", defName = "_0_def", elements = [{name = "_1", tpe = {class = "sifive.enterprise.grandcentral.AugmentedGroundType", ref = {circuit = "GCTInterface", component = [{class = "firrtl.annotations.TargetToken$Field", value = "_0"}, {class = "firrtl.annotations.TargetToken$Field", value = "_1"}], module = "GCTInterface", path = [], ref = "r"}, tpe = {class = "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"}}}, {name = "_0", tpe = {class = "sifive.enterprise.grandcentral.AugmentedGroundType", ref = {circuit = "GCTInterface", component = [{class = "firrtl.annotations.TargetToken$Field", value = "_0"}, {class = "firrtl.annotations.TargetToken$Field", value = "_0"}], module = "GCTInterface", path = [], ref = "r"}, tpe = {class = "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"}}}]}}]}}, {description = "the port 'a' in GCTInterface", name = "port", tpe = {class = "sifive.enterprise.grandcentral.AugmentedGroundType", ref = {circuit = "GCTInterface", component = [], module = "GCTInterface", path = [], ref = "a"}, tpe = {class = "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"}}}]}}]} {
  firrtl.module private @view_companion() {
    firrtl.skip
  }
  firrtl.module @GCTInterface(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %a: !firrtl.uint<1>) {
    %r = firrtl.reg %clock  : !firrtl.bundle<_0: bundle<_0: uint<1>, _1: uint<1>>, _2: vector<uint<1>, 2>>
    firrtl.instance view_companion  @view_companion()
  }
}

// CHECK-LABEL: firrtl.circuit "GCTInterface"

// The interface definition should show up as a circuit annotation.  Nested
// interfaces show up as nested bundle types and not as separate interfaces.
// CHECK-SAME: annotations
// CHECK-SAME: {unrelatedAnnotation}
// CHECK-SAME: {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
// CHECK-SAME:  defName = "ViewName",
// CHECK-SAME:  elements = [
// CHECK-SAME:    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
// CHECK-SAME:     defName = "Register",
// CHECK-SAME:     description = "the register in GCTInterface",
// CHECK-SAME:     elements = [
// CHECK-SAME:       {class = "sifive.enterprise.grandcentral.AugmentedVectorType",
// CHECK-SAME:        elements = [
// CHECK-SAME:          {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
// CHECK-SAME:           id = [[ID_2_0:[0-9]+]] : i64,
// CHECK-SAME:           name = "_2"},
// CHECK-SAME:          {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
// CHECK-SAME:           id = [[ID_2_1:[0-9]+]] : i64,
// CHECK-SAME:           name = "_2"}],
// CHECK-SAME:        name = "_2"},
// CHECK-SAME:       {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
// CHECK-SAME:        defName = "_0_def",
// CHECK-SAME:        elements = [
// CHECK-SAME:          {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
// CHECK-SAME:           id = [[ID_1:[0-9]+]] : i64,
// CHECK-SAME:           name = "_1"},
// CHECK-SAME:          {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
// CHECK-SAME:           id = [[ID_0:[0-9]+]] : i64,
// CHECK-SAME:           name = "_0"}],
// CHECK-SAME:        name = "_0_inst"}],
// CHECK-SAME:     name = "register"},
// CHECK-SAME:    {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
// CHECK-SAME:     description = "the port 'a' in GCTInterface",
// CHECK-SAME:     id = [[ID_port:[0-9]+]] : i64,
// CHECK-SAME:     name = "port"}],
// CHECK-SAME:  id = [[ID_ViewName:[0-9]+]] : i64,
// CHECK-SAME:  name = "view"}

// The companion should be marked.
// CHECK: firrtl.module private @view_companion
// CHECK-SAME: annotations
// CHECK-SAME: {class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
// CHECK-SAME:  id = [[ID_ViewName]] : i64,
// CHECK-SAME:  type = "companion"}

// The parent should be annotated. Additionally, this example has all the
// members of the interface inside the parent.  Both port "a" and register
// "r" should be annotated.
// CHECK: firrtl.module @GCTInterface
// CHECK-SAME: %a: !firrtl.uint<1> sym @a [
// CHECK-SAME:   {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
// CHECK-SAME:    d = [[ID_port]] : i64}
// CHECK-SAME: annotations = [
// CHECK-SAME:   {class = "sifive.enterprise.grandcentral.ViewAnnotation.parent",
// CHECK-SAME:    id = [[ID_ViewName]] : i64,
// CHECK-SAME:    name = "view",
// CHECK-SAME:    type = "parent"}]
// CHECK: firrtl.reg
// CHECK-SAME: annotations
// CHECK-SAME:   {circt.fieldID = 2 : i32,
// CHECK-SAME:    class = "sifive.enterprise.grandcentral.AugmentedGroundType",
// CHECK-SAME:    id = [[ID_0]] : i64}
// CHECK-SAME:   {circt.fieldID = 3 : i32,
// CHECK-SAME:    class = "sifive.enterprise.grandcentral.AugmentedGroundType",
// CHECK-SAME:    id = [[ID_1]] : i64}
// CHECK-SAME:   {circt.fieldID = 6 : i32,
// CHECK-SAME:    class = "sifive.enterprise.grandcentral.AugmentedGroundType",
// CHECK-SAME:    id = [[ID_2_1]] : i64}
// CHECK-SAME:   {circt.fieldID = 5 : i32,
// CHECK-SAME:    class = "sifive.enterprise.grandcentral.AugmentedGroundType",
// CHECK-SAME:    id = [[ID_2_0]] : i64}

// -----

firrtl.circuit "Foo"  attributes {rawAnnotations = [{class = "sifive.enterprise.grandcentral.ViewAnnotation", companion = "~Foo|Bar_companion", name = "Bar", parent = "~Foo|Foo", view = {class = "sifive.enterprise.grandcentral.AugmentedBundleType", defName = "View", elements = [{description = "a string", name = "string", tpe = {class = "sifive.enterprise.grandcentral.AugmentedStringType", value = "hello"}}, {description = "a boolean", name = "boolean", tpe = {class = "sifive.enterprise.grandcentral.AugmentedBooleanType", value = false}}, {description = "an integer", name = "integer", tpe = {class = "sifive.enterprise.grandcentral.AugmentedIntegerType", value = 42 : i64}}, {description = "a double", name = "double", tpe = {class = "sifive.enterprise.grandcentral.AugmentedDoubleType", value = 3.140000e+00 : f64}}]}}]} {
  firrtl.extmodule private @Bar_companion()
  firrtl.module @Foo() {
     firrtl.instance Bar_companion @Bar_companion()
   }
}

// CHECK-LABEL: firrtl.circuit "Foo"
// CHECK-SAME: annotations = [{class = "[[_:.+]]AugmentedBundleType", [[_:.+]] elements = [{
// CHECK-SAME: "sifive.enterprise.grandcentral.AugmentedStringType"
// CHECK-SAME: "sifive.enterprise.grandcentral.AugmentedBooleanType"
// CHECK-SAME: "sifive.enterprise.grandcentral.AugmentedIntegerType"
// CHECK-SAME: "sifive.enterprise.grandcentral.AugmentedDoubleType"
