// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-lower-annotations)' --split-input-file %s | FileCheck %s

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
// CHECK: firrtl.hierpath @nla_1 [@FooNL::@baz, @BazNL::@bar, @BarNL]
// CHECK: firrtl.hierpath @nla_0 [@FooNL::@baz, @BazNL::@bar, @BarNL::@w]
// CHECK: firrtl.hierpath @nla [@FooNL::@baz, @BazNL::@bar, @BarNL::@w2]
// CHECK: firrtl.module @BarNL
// CHECK: %w = firrtl.wire sym @w {annotations = [{circt.nonlocal = @nla_0, class = "circt.test", nl = "nl"}]}
// CHECK: %w2 = firrtl.wire sym @w2 {annotations = [{circt.fieldID = 5 : i32, circt.nonlocal = @nla, class = "circt.test", nl = "nl2"}]} : !firrtl.bundle<a: uint, b: vector<uint, 4>>
// CHECK: firrtl.instance bar sym @bar @BarNL()
// CHECK: firrtl.instance baz sym @baz @BazNL()
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
// CHECK: firrtl.hierpath @nla [@MemPortsNL::@child, @Child::@bar]
// CHECK: firrtl.module @Child()
// CHECK:   %bar_r = firrtl.mem sym @bar
// CHECK-SAME: portAnnotations = {{\[}}[{circt.nonlocal = @nla, class = "circt.test", nl = "nl"}]]
// CHECK: firrtl.module @MemPortsNL()
// CHECK:   firrtl.instance child sym @child
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
  // CHECK: firrtl.hierpath @nla [@Test::@exttest, @ExtTest]
  // CHECK: firrtl.extmodule @ExtTest() attributes {annotations = [{circt.nonlocal = @nla, class = "circt.test"}]}
  firrtl.extmodule @ExtTest()

  firrtl.module @Test() {
    // CHECK: firrtl.instance exttest sym @exttest @ExtTest()
    firrtl.instance exttest @ExtTest()
  }
}

// -----

// Annotations on instances should be moved to the target module.
firrtl.circuit "Test" attributes {rawAnnotations = [
  {class = "circt.test", target = "~Test|Test>exttest.in"}
  ]} {
  // CHECK: firrtl.hierpath @nla [@Test::@exttest, @ExtTest::@in]
  // CHECK: firrtl.extmodule @ExtTest(in in: !firrtl.uint<1> sym @in [{circt.nonlocal = @nla, class = "circt.test"}])
  firrtl.extmodule @ExtTest(in in: !firrtl.uint<1>)

  firrtl.module @Test() {
    // CHECK: %exttest_in = firrtl.instance exttest sym @exttest @ExtTest(in in: !firrtl.uint<1>)
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

// -----

// SiFive-custom annotations related to the GrandCentral utility.  These
// annotations do not conform to standard SingleTarget or NoTarget format and
// need to be manually split up.

// Test sifive.enterprise.grandcentral.DataTapsAnnotation with all possible
// variants of DataTapKeys.

firrtl.circuit "GCTDataTap" attributes {rawAnnotations = [{
  blackBox = "~GCTDataTap|DataTap",
  class = "sifive.enterprise.grandcentral.DataTapsAnnotation",
  keys = [
    {
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
      portName = "~GCTDataTap|DataTap>_0",
      source = "~GCTDataTap|GCTDataTap>r"
    },
    {
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
      portName = "~GCTDataTap|DataTap>_1[0]",
      source = "~GCTDataTap|GCTDataTap>r"
    },
    {
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
      portName = "~GCTDataTap|DataTap>_2",
      source = "~GCTDataTap|GCTDataTap>w.a"
    },
    {
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
      portName = "~GCTDataTap|DataTap>_3[0]",
      source = "~GCTDataTap|GCTDataTap>w.a"
    },
    {
      class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey",
      internalPath = "baz.qux",
      module = "~GCTDataTap|BlackBox",
      portName = "~GCTDataTap|DataTap>_4"
    },
    {
      class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey",
      internalPath = "baz.quz",
      module = "~GCTDataTap|BlackBox",
      portName = "~GCTDataTap|DataTap>_5[0]"
    },
    {
      class = "sifive.enterprise.grandcentral.DeletedDataTapKey",
      portName = "~GCTDataTap|DataTap>_6"
    },
    {
      class = "sifive.enterprise.grandcentral.DeletedDataTapKey",
      portName = "~GCTDataTap|DataTap>_7[0]"
    },
    {
      class = "sifive.enterprise.grandcentral.LiteralDataTapKey",
      literal = "UInt<16>(\22h2a\22)",
      portName = "~GCTDataTap|DataTap>_8"
    },
    {
      class = "sifive.enterprise.grandcentral.LiteralDataTapKey",
      literal = "UInt<16>(\22h2a\22)",
      portName = "~GCTDataTap|DataTap>_9[0]"
    },
    {
      class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
      portName = "~GCTDataTap|DataTap>_10",
      source = "~GCTDataTap|GCTDataTap/im:InnerMod>w"
    }
  ]
}]} {
  firrtl.extmodule private @DataTap(
    out _0: !firrtl.uint<1>,
    out _1: !firrtl.vector<uint<1>, 1>,
    out _2: !firrtl.uint<1>,
    out _3: !firrtl.vector<uint<1>, 1>,
    out _4: !firrtl.uint<1>,
    out _5: !firrtl.vector<uint<1>, 1>,
    out _6: !firrtl.uint<1>,
    out _7: !firrtl.vector<uint<1>, 1>,
    out _8: !firrtl.uint<1>,
    out _9: !firrtl.vector<uint<1>, 1>,
    out _10: !firrtl.uint<1>
  ) attributes {defname = "DataTap"}
  firrtl.extmodule private @BlackBox() attributes {defname = "BlackBox"}
  firrtl.module private @InnerMod() {
    %w = firrtl.wire : !firrtl.uint<1>
  }
  firrtl.module @GCTDataTap(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    %r = firrtl.reg %clock  : !firrtl.uint<1>
    %w = firrtl.wire : !firrtl.bundle<a: uint<1>>
    %DataTap__0, %DataTap__1, %DataTap__2, %DataTap__3, %DataTap__4, %DataTap__5, %DataTap__6, %DataTap__7, %DataTap__8, %DataTap__9, %DataTap__10 = firrtl.instance DataTap  @DataTap(out _0: !firrtl.uint<1>, out _1: !firrtl.vector<uint<1>, 1>, out _2: !firrtl.uint<1>, out _3: !firrtl.vector<uint<1>, 1>, out _4: !firrtl.uint<1>, out _5: !firrtl.vector<uint<1>, 1>, out _6: !firrtl.uint<1>, out _7: !firrtl.vector<uint<1>, 1>, out _8: !firrtl.uint<1>, out _9: !firrtl.vector<uint<1>, 1>, out _10: !firrtl.uint<1>)
    firrtl.instance BlackBox @BlackBox()
    firrtl.instance im @InnerMod()
  }
}

// CHECK-LABEL: firrtl.circuit "GCTDataTap"
// CHECK:      firrtl.hierpath [[NLA:@.+]] [@GCTDataTap::@im, @InnerMod::@w]

// CHECK-LABEL: firrtl.extmodule private @DataTap

// CHECK-SAME: _0: !firrtl.uint<1>
// CHECK-SAME:   class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.port"
// CHECK-SAME:   id = [[ID:[0-9]+]] : i64
// CHECK-SAME:   portID = [[PORT_ID_0:[0-9]+]] : i64

// CHECK-SAME: _1: !firrtl.vector<uint<1>, 1>
// CHECK-SAME:   circt.fieldID = 1
// CHECK-SAME:   class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.port"
// CHECK-SAME:   id = [[ID]] : i64
// CHECK-SAME:   portID = [[PORT_ID_1:[0-9]+]] : i64

// CHECK-SAME: _2: !firrtl.uint<1>
// CHECK-SAME:   class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.port"
// CHECK-SAME:   id = [[ID]] : i64
// CHECK-SAME:   portID = [[PORT_ID_2:[0-9]+]] : i64

// CHECK-SAME: _3: !firrtl.vector<uint<1>, 1> [
// CHECK-SAME:   circt.fieldID = 1
// CHECK-SAME:   class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.port"
// CHECK-SAME:   id = [[ID]] : i64
// CHECK-SAME:   portID = [[PORT_ID_3:[0-9]+]] : i64

// CHECK-SAME: _4: !firrtl.uint<1>
// CHECK-SAME:   class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey.port"
// CHECK-SAME:   id = [[ID]] : i64
// CHECK-SAME:   portID = [[PORT_ID_4:[0-9]+]] : i64

// CHECK-SAME: _5: !firrtl.vector<uint<1>, 1>
// CHECK-SAME:   circt.fieldID = 1
// CHECK-SAME:   class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey.port"
// CHECK-SAME:   id = [[ID]] : i64
// CHECK-SAME:   portID = [[PORT_ID_5:[0-9]+]] : i64

// CHECK-SAME: _6: !firrtl.uint<1>
// CHECK-SAME:   class = "sifive.enterprise.grandcentral.DeletedDataTapKey"
// CHECK-SAME:   id = [[ID]] : i64

// CHECK-SAME: _7: !firrtl.vector<uint<1>, 1>
// CHECK-SAME:   circt.fieldID = 1
// CHECK-SAME:   class = "sifive.enterprise.grandcentral.DeletedDataTapKey"
// CHECK-SAME:   id = [[ID]] : i64

// CHECK-SAME: _8: !firrtl.uint<1>
// CHECK-SAME:   class = "sifive.enterprise.grandcentral.LiteralDataTapKey"
// CHECK-SAME:   literal = "UInt<16>(\22h2a\22)"

// CHECK-SAME: _9: !firrtl.vector<uint<1>, 1>
// CHECK-SAME:   circt.fieldID = 1
// CHECK-SAME:   class = "sifive.enterprise.grandcentral.LiteralDataTapKey"
// CHECK-SAME:   literal = "UInt<16>(\22h2a\22)"

// CHECK-SAME: _10: !firrtl.uint<1>
// CHECK-SAME:   class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.port"
// CHECK-SAME:   id = [[ID]] : i64
// CHECK-SAME:   portID = [[PORT_ID_6:[0-9]+]] : i64

// CHECK-SAME: annotations = [
// CHECK-SAME:   {class = "sifive.enterprise.grandcentral.DataTapsAnnotation.blackbox"}
// CHECK-SAME: ]

// CHECK-LABEL: firrtl.extmodule private @BlackBox
// CHECK-SAME: annotations = [
// CHECK-SAME:   {
// CHECK-SAME:     class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey.source",
// CHECK-SAME:     id = [[ID]]
// CHECK-SAME:     internalPath = "baz.quz",
// CHECK-SAME:     portID = [[PORT_ID_5]] : i64
// CHECK-SAME:   }
// CHECK-SAME:   {
// CHECK-SAME:     class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey.source",
// CHECK-SAME:     id = [[ID]]
// CHECK-SAME:     internalPath = "baz.qux",
// CHECK-SAME:     portID = [[PORT_ID_4]] : i64
// CHECK-SAME:   }
// CHECK-SAME: ]

// CHECK-LABEL: firrtl.module private @InnerMod
// CHECK-NEXT: %w = firrtl.wire sym @w
// CHECK-SAME: annotations = [
// CHECK-SAME:   {
// CHECK-SAME:     circt.nonlocal = [[NLA]]
// CHECK-SAME:     class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.source"
// CHECK-SAME:     id = [[ID]]
// CHECK-SAME:     portID = [[PORT_ID_6]]
// CHECK-SAME:   }
// CHECK-SAME: ]

// CHECK: firrtl.module @GCTDataTap
// CHECK-LABEL: firrtl.reg sym @r
// CHECk-SAME: annotations = [
// CHECK-SAME:   {
// CHECK-SAME:     class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.source"
// CHECK-SAME:     id = [[ID]]
// CHECK-SAME:     portID = [[PORT_ID_0]]
// CHECK-SAME:   }
// CHECK-SAME: ]

// CHECK-LABEL: firrtl.wire
// CHECK-SAME: annotations = [
// CHECK-SAME:   {
// CHECK-SAME:     circt.fieldID = 1
// CHECK-SAME:     class = "firrtl.transforms.DontTouchAnnotation"
// CHECK-SAME:   }
// CHECK-SAME:   {
// CHECK-SAME:     circt.fieldID = 1
// CHECK-SAME:     class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.source"
// CHECK-SAME:     id = [[ID]]
// CHECK-SAME:     portID = [[PORT_ID_3]]
// CHECK-SAME:   }
// CHECK-SAME:   {
// CHECK-SAME:     class = "sifive.enterprise.grandcentral.ReferenceDataTapKey.source",
// CHECK-SAME:      id = [[ID]]
// CHECK-SAME:      portID = [[PORT_ID_2]]
// CHECK-SAME:   }
// CHECK-SAME: ]

// -----

// Test sifive.enterprise.grandcentral.MemTapAnnotation
firrtl.circuit "GCTMemTap" attributes {rawAnnotations = [{
  class = "sifive.enterprise.grandcentral.MemTapAnnotation",
  source = "~GCTMemTap|GCTMemTap>mem",
  taps = ["GCTMemTap.MemTap.mem[0]", "GCTMemTap.MemTap.mem[1]"]
}]} {
  firrtl.extmodule private @MemTap(out mem: !firrtl.vector<uint<1>, 2>) attributes {defname = "MemTap"}
  firrtl.module @GCTMemTap(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
    %mem = chirrtl.combmem  : !chirrtl.cmemory<uint<1>, 2>
    %MemTap_mem = firrtl.instance MemTap  @MemTap(out mem: !firrtl.vector<uint<1>, 2>)
    %0 = firrtl.subindex %MemTap_mem[1] : !firrtl.vector<uint<1>, 2>
    %1 = firrtl.subindex %MemTap_mem[0] : !firrtl.vector<uint<1>, 2>
    %memTap = firrtl.wire  : !firrtl.vector<uint<1>, 2>
    %2 = firrtl.subindex %memTap[1] : !firrtl.vector<uint<1>, 2>
    %3 = firrtl.subindex %memTap[0] : !firrtl.vector<uint<1>, 2>
    firrtl.strictconnect %3, %1 : !firrtl.uint<1>
    firrtl.strictconnect %2, %0 : !firrtl.uint<1>
  }
}


// CHECK-LABEL: firrtl.circuit "GCTMemTap"

// CHECK-LABEL: firrtl.extmodule private @MemTap
// CHECK-SAME: mem: !firrtl.vector<uint<1>, 2> [
// CHECK-SAME:   {
// CHECK-SAME:     circt.fieldID = 2
// CHECK-SAME:     class = "sifive.enterprise.grandcentral.MemTapAnnotation.port"
// CHECK-SAME:     id = [[ID]]
// CHECK-SAME:     portID = 1
// CHECK-SAME:   }
// CHECK-SAME:   {
// CHECK-SAME:     circt.fieldID = 1
// CHECK-SAME:     class = "sifive.enterprise.grandcentral.MemTapAnnotation.port"
// CHECK-SAME:     id = [[ID:[0-9]+]] : i64
// CHECK-SAME:     portID = 0
// CHECK-SAME:   }

// CHECK-LABEL: firrtl.module @GCTMemTap
// CHECK: %mem = chirrtl.combmem
// CHECK-SAME: annotations = [
// CHECK-SAME:   {
// CHECK-SAME:     class = "sifive.enterprise.grandcentral.MemTapAnnotation.source"
// CHECK-SAME:     id = [[ID]]
// CHECK-SAME:   }
// CHECK-SAME: ]
