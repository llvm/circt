// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-lower-annotations)' -split-input-file %s |FileCheck %s

// circt.test copies the annotation to the target
// circt.testNT puts the targetless annotation on the circuit


// CHECK-LABEL: firrtl.circuit "Foo1" attributes {annotations = [{a = "a", class = "circt.testNT"}]}
firrtl.circuit "Foo1"  attributes {annotations = [{a = "a", class = "circt.testNT"}]}  {
  firrtl.module @Foo1() {
    firrtl.skip
  }
}

// -----

// A legacy `firrtl.annotations.CircuitName` annotation becomes a CircuitTarget Annotation.

// CHECK-LABEL: firrtl.circuit "Foo2" attributes {annotations = [{a = "a", class = "circt.test"}]}
firrtl.circuit "Foo2"  attributes {annotations = [{a = "a", class = "circt.test", target = "Foo2"}]}  {
  firrtl.module @Foo2() {
    firrtl.skip
  }
}

// -----

// A CircuitTarget Annotation is attached to the circuit.

// CHECK-LABEL: firrtl.circuit "Foo3" attributes {annotations = [{a = "a", class = "circt.test"}]}
firrtl.circuit "Foo3"  attributes {annotations = [{a = "a", class = "circt.test", target = "~Foo3"}]}  {
  firrtl.module @Foo3() {
    firrtl.skip
  }
}

// -----

// A legacy `firrtl.annotations.ModuleName` annotation becomes a ModuleTarget
// Annotation

// CHECK-LABEL: firrtl.module @Foo4() attributes {annotations = [{a = "a", class = "circt.test"}]}
firrtl.circuit "Foo4"  attributes {annotations = [{a = "a", class = "circt.test", target = "Foo4.Foo4"}]}  {
  firrtl.module @Foo4() {
    firrtl.skip
  }
}

// -----

// A ModuleTarget Annotation is attached to the correct module.

// CHECK-LABEL: firrtl.module @Foo5() attributes {annotations = [{a = "a", class = "circt.test"}]}
firrtl.circuit "Foo5"  attributes {annotations = [{a = "a", class = "circt.test", target = "~Foo5|Foo5"}]}  {
  firrtl.module @Foo5() {
    firrtl.skip
  }
}

// -----

// A ModuleTarget Annotation can be attached to an ExtModule.

// CHECK-LABEL: firrtl.extmodule @Bar6
// CHECK-SAME: attributes {annotations = [{a = "a", class = "circt.test"}]}
firrtl.circuit "Foo6"  attributes {annotations = [{a = "a", class = "circt.test", target = "~Foo6|Bar6"}]}  {
  firrtl.extmodule @Bar6(in %a: !firrtl.uint<1>)
  firrtl.module @Foo6(in %a: !firrtl.uint<1>) {
    %bar_a = firrtl.instance @Bar6  {name = "bar"} : !firrtl.uint<1>
    firrtl.connect %bar_a, %a : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// A ReferenceTarget, ComponentName, or InstanceTarget pointing at an Instance
// should work.

// CHECK-LABEL: firrtl.instance @BarI
// CHECK-SAME: annotations = [{a = "a", class = "circt.test"}, {b = "b", class = "circt.test"}, {c = "c", class = "circt.test"}]
firrtl.circuit "FooI"  attributes {annotations = [{a = "a", class = "circt.test", target = "~FooI|FooI>bar"}, {b = "b", class = "circt.test", target = "FooI.FooI.bar"}, {c = "c", class = "circt.test", target = "~FooI|FooI/bar:BarI"}]}  {
  firrtl.module @BarI() {
    firrtl.skip
  }
  firrtl.module @FooI() {
    firrtl.instance @BarI  {name = "bar"}
  }
}

// -----

// A non-local annotation should work.

// CHECK-LABEL: firrtl.circuit "FooNL"
// CHECK: firrtl.module @BarNL
// CHECK: firrtl.wire {annotations = [{circt.nonlocal = @"~FooNL|FooNL/baz:BazNL/bar:BarNL>w_NA_0", class = "circt.test", nl = "nl"}]}
// CHECK: firrtl.instance @BarNL {annotations = [{circt.nonlocal = @"~FooNL|FooNL/baz:BazNL/bar:BarNL>w_NA_0", class = "circt.nonlocal"}], name = "bar"}
// CHECK: firrtl.instance @BazNL {annotations = [{circt.nonlocal = @"~FooNL|FooNL/baz:BazNL/bar:BarNL>w_NA_0", class = "circt.nonlocal"}], name = "baz"
// CHECK: firrtl.nla @"~FooNL|FooNL/baz:BazNL/bar:BarNL>w_NA_0"
firrtl.circuit "FooNL"  attributes {annotations = [{class = "circt.test", nl = "nl", target = "~FooNL|FooNL/baz:BazNL/bar:BarNL>w"}]}  {
  firrtl.module @BarNL() {
    %w = firrtl.wire  : !firrtl.uint
    firrtl.skip
  }
  firrtl.module @BazNL() {
    firrtl.instance @BarNL  {name = "bar"}
  }
  firrtl.module @FooNL() {
    firrtl.instance @BazNL  {name = "baz"}
  }
}

// -----

// Test result annotations of InstanceOp.

// CHECK-LABEL: firrtl.circuit "FooIR"
// CHECK: %bar_a, %bar_b, %bar_c = firrtl.instance @BarIR
// CHECK-SAME: {class = "circt.test", one}
// CHECK-SAME: {circt.fieldID = 1 : i32, class = "circt.test", two}
// CHECK-SAME: {class = "circt.test", four}
firrtl.circuit "FooIR"  attributes {annotations = [{class = "circt.test", one, target = "~FooIR|FooIR>bar.a"}, {class = "circt.test", target = "~FooIR|FooIR>bar.b.baz", two}, {class = "circt.test", four, target = "FooIR.FooIR.bar.c"}]}  {
  firrtl.module @BarIR(in %a: !firrtl.uint<1>, out %b: !firrtl.bundle<baz: uint<1>, qux: uint<1>>, out %c: !firrtl.uint<1>) {
  }
  firrtl.module @FooIR() {
    %bar_a, %bar_b, %bar_c = firrtl.instance @BarIR  {name = "bar"} : !firrtl.uint<1>, !firrtl.bundle<baz: uint<1>, qux: uint<1>>, !firrtl.uint<1>
  }
}

// -----

// A ReferenceTarget/ComponentName pointing at a CombMem should work.

// CHECK-LABEL: firrtl.module @Foo7
// CHECK: firrtl.combmem
// CHECK-SAME: annotations = [{a = "a", class = "circt.test"}, {b = "b", class = "circt.test"}]
firrtl.circuit "Foo7"  attributes {annotations = [{a = "a", class = "circt.test", target = "~Foo7|Foo7>bar"}, {b = "b", class = "circt.test", target = "Foo7.Foo7.bar"}]}  {
  firrtl.module @Foo7() {
    %bar = firrtl.combmem  {name = "bar"} : !firrtl.cmemory<uint<1>, 8>
  }
}

// -----

// A ReferenceTarget/ComponentName pointing at a memory should work.

// CHECK-LABEL: module {
// CHECK: firrtl.mem
// CHECK-SAME: annotations = [{a = "a", class = "circt.test"}, {b = "b", class = "circt.test"}]
firrtl.circuit "Foo"  attributes {annotations = [{a = "a", class = "circt.test", target = "~Foo|Foo>bar"}, {b = "b", class = "circt.test", target = "Foo.Foo.bar"}]}  {
  firrtl.module @Foo() {
    %bar_r, %bar_w = firrtl.mem Undefined  {depth = 16 : i64, name = "bar", portNames = ["r", "w"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
  }
}

// -----

// Test result annotations of MemOp.

// CHECK-LABEL: module {
// CHECK: firrtl.mem
// CHECK-SAME: portAnnotations = [
// CHECK-SAME: [{a, class = "circt.test"}, {b, circt.fieldID = 5 : i32, class = "circt.test"}]
// CHECK-SAME: [{c, circt.fieldID = 2 : i32, class = "circt.test"}, {circt.fieldID = 6 : i32, class = "circt.test", d}]
firrtl.circuit "Foo"  attributes {annotations = [{a, class = "circt.test", target = "~Foo|Foo>bar.r"}, {b, class = "circt.test", target = "~Foo|Foo>bar.r.data.baz"}, {c, class = "circt.test", target = "~Foo|Foo>bar.w.en"}, {class = "circt.test", d, target = "~Foo|Foo>bar.w.data.qux"}]}  {
  firrtl.module @Foo() {
    %bar_r, %bar_w = firrtl.mem Undefined  {depth = 16 : i64, name = "bar", portNames = ["r", "w"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<baz: uint<8>, qux: uint<8>>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<baz: uint<8>, qux: uint<8>>, mask: bundle<baz: uint<1>, qux: uint<1>>>
  }
}

// -----

// A ReferenceTarget/ComponentName pointing at a node should work.  This
// shouldn't crash if the node is in a nested block.

// CHECK-LABEL: module {
// CHECK: %bar = firrtl.node
// CHECK-SAME: annotations = [{a = "a", class = "circt.test"}
// CHECK: %baz = firrtl.node
// CHECK-SAME: annotations = [{b = "b", class = "circt.test"}]
firrtl.circuit "Foo"  attributes {annotations = [{a = "a", class = "circt.test", target = "~Foo|Foo>bar"}, {b = "b", class = "circt.test", target = "Foo.Foo.baz"}]}  {
  firrtl.module @Foo(in %cond: !firrtl.vector<uint<1>, 2>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %bar = firrtl.node %c0_ui1  : !firrtl.uint<1>
    %0 = firrtl.subindex %cond[0] : !firrtl.vector<uint<1>, 2>
    firrtl.when %0  {
      %1 = firrtl.subindex %cond[1] : !firrtl.vector<uint<1>, 2>
      firrtl.when %1  {
        %baz = firrtl.node %c0_ui1  : !firrtl.uint<1>
      }
    }
  }
}

// -----
// A ReferenceTarget/ComponentName pointing at a wire should work.

// CHECK-LABEL: module {
// CHECK: %bar = firrtl.wire
// CHECK-SAME: annotations = [{a = "a", class = "circt.test"}, {b = "b", class = "circt.test"}]
firrtl.circuit "Foo"  attributes {annotations = [{a = "a", class = "circt.test", target = "~Foo|Foo>bar"}, {b = "b", class = "circt.test", target = "Foo.Foo.bar"}]}  {
  firrtl.module @Foo() {
    %bar = firrtl.wire  : !firrtl.uint<1>
  }
}

// -----
// A ReferenceTarget/ComponentName pointing at a register should work.

// CHECK-LABEL: module {
// CHECK: %bar = firrtl.reg
// CHECK-SAME: annotations = [{a = "a", class = "circt.test"}]
// CHECK: %baz = firrtl.regreset
// CHECK-SAME: annotations = [{b = "b", class = "circt.test"}]
firrtl.circuit "Foo"  attributes {annotations = [{a = "a", class = "circt.test", target = "~Foo|Foo>bar"}, {b = "b", class = "circt.test", target = "Foo.Foo.baz"}]}  {
  firrtl.module @Foo(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
    %bar = firrtl.reg %clock  : !firrtl.uint<1>
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %baz = firrtl.regreset %clock, %reset, %c0_ui1  : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// A ReferenceTarget/ComponentName pointing at an SMem should work.

// CHECK-LABEL: firrtl.seqmem
// CHECK-SAME: annotations = [{a = "a", class = "circt.test"}, {b = "b", class = "circt.test"}]
firrtl.circuit "Foo"  attributes {annotations = [{a = "a", class = "circt.test", target = "~Foo|Foo>bar"}, {b = "b", class = "circt.test", target = "Foo.Foo.bar"}]}  {
  firrtl.module @Foo() {
    %bar = firrtl.seqmem Undefined  {name = "bar"} : !firrtl.cmemory<uint<1>, 8>
  }
}

// -----

// A ReferenceTarget/ComponentName pointing at a module/extmodule port should work.

// CHECK-LABEL: firrtl.extmodule @Bar
// CHECK-SAME: portAnnotations
// CHECK-SAME: a = "a", class = "circt.test"
// CHECK: firrtl.module @Foo
// CHECK-SAME: portAnnotations
// CHECK-SAME: b = "b", class = "circt.test"
firrtl.circuit "Foo"  attributes {annotations = [{a = "a", class = "circt.test", target = "~Foo|Bar>bar"}, {b = "b", class = "circt.test", target = "Foo.Foo.foo"}]}  {
  firrtl.extmodule @Bar(in %bar: !firrtl.uint<1>)
  firrtl.module @Foo(in %foo: !firrtl.uint<1>) {
    %bar_bar = firrtl.instance @Bar  {name = "bar"} : !firrtl.uint<1>
    firrtl.connect %bar_bar, %foo : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// Subfield/Subindex annotations should be parsed correctly on wires

// CHECK-LABEL: %bar = firrtl.wire  {annotations =
// CHECK-SAME: {circt.fieldID = 1 : i32, class = "circt.test", one}
// CHECK-SAME: {circt.fieldID = 5 : i32, class = "circt.test", two}
firrtl.circuit "Foo"  attributes {annotations = [{class = "circt.test", one, target = "~Foo|Foo>bar[0]"}, {class = "circt.test", target = "~Foo|Foo>bar[1].baz", two}]}  {
  firrtl.module @Foo() {
    %bar = firrtl.wire  : !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>
  }
}

// -----

// Subfield/Subindex annotations should be parsed correctly on registers

// CHECK-LABEL: %bar = firrtl.reg %clock  {annotations =
// CHECK-SAME: {circt.fieldID = 1 : i32, class = "circt.test", one}
// CHECK-SAME: {circt.fieldID = 5 : i32, class = "circt.test", two}
firrtl.circuit "Foo"  attributes {annotations = [{class = "circt.test", one, target = "~Foo|Foo>bar[0]"}, {class = "circt.test", target = "~Foo|Foo>bar[1].baz", two}]}  {
  firrtl.module @Foo(in %clock: !firrtl.clock) {
    %bar = firrtl.reg %clock  : !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>
  }
}

// -----

// Subindices should not get sign-extended and cause problems.  This circuit has
// caused bugs in the past.

// CHECK-LABEL: %w = firrtl.wire {annotations =
// CHECK-SAME: {a, circt.fieldID = 10 : i32, class = "circt.test"}
firrtl.circuit "Foo"  attributes {annotations = [{a, class = "circt.test", target = "~Foo|Foo>w[9]"}]}  {
  firrtl.module @Foo(in %a: !firrtl.vector<uint<1>, 18>, out %b: !firrtl.vector<uint<1>, 18>) {
    %w = firrtl.wire  : !firrtl.vector<uint<1>, 18>
    firrtl.connect %w, %a : !firrtl.vector<uint<1>, 18>, !firrtl.vector<uint<1>, 18>
    firrtl.connect %b, %w : !firrtl.vector<uint<1>, 18>, !firrtl.vector<uint<1>, 18>
  }
}

// -----

// 
// GrandCentral annotation lowering
// 
// SiFive-custom annotations related to the GrandCentral utility.  These
// annotations do not conform to standard SingleTarget or NoTarget format and
// need to be manually split up.
// 

  firrtl.circuit "GCTDataTap"  attributes {annotations = [{blackBox = "~GCTDataTap|DataTap", class = "sifive.enterprise.grandcentral.DataTapsAnnotation", keys = [{class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", portName = "~GCTDataTap|DataTap>_0", source = "~GCTDataTap|GCTDataTap>r"}, {class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", portName = "~GCTDataTap|DataTap>_1[0]", source = "~GCTDataTap|GCTDataTap>r"}, {class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", portName = "~GCTDataTap|DataTap>_2", source = "~GCTDataTap|GCTDataTap>w.a"}, {class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", portName = "~GCTDataTap|DataTap>_3[0]", source = "~GCTDataTap|GCTDataTap>w.a"}, {class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey", internalPath = "baz.qux", module = "~GCTDataTap|BlackBox", portName = "~GCTDataTap|DataTap>_4"}, {class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey", internalPath = "baz.quz", module = "~GCTDataTap|BlackBox", portName = "~GCTDataTap|DataTap>_5[0]"}, {class = "sifive.enterprise.grandcentral.DeletedDataTapKey", portName = "~GCTDataTap|DataTap>_6"}, {class = "sifive.enterprise.grandcentral.DeletedDataTapKey", portName = "~GCTDataTap|DataTap>_7[0]"}, {class = "sifive.enterprise.grandcentral.LiteralDataTapKey", literal = "UInt<16>(\22h2a\22)", portName = "~GCTDataTap|DataTap>_8"}, {class = "sifive.enterprise.grandcentral.LiteralDataTapKey", literal = "UInt<16>(\22h2a\22)", portName = "~GCTDataTap|DataTap>_9[0]"}]}, {class = "circt.testNT", unrelatedAnnotation}]}  {
    firrtl.extmodule @DataTap(out %_0: !firrtl.uint<1>, out %_1: !firrtl.vector<uint<1>, 1>, out %_2: !firrtl.uint<1>, out %_3: !firrtl.vector<uint<1>, 1>, out %_4: !firrtl.uint<1>, out %_5: !firrtl.vector<uint<1>, 1>, out %_6: !firrtl.uint<1>, out %_7: !firrtl.vector<uint<1>, 1>, out %_8: !firrtl.uint<1>, out %_9: !firrtl.vector<uint<1>, 1>) attributes {defname = "DataTap"}
    firrtl.extmodule @BlackBox() attributes {defname = "BlackBox"}
    firrtl.module @GCTDataTap(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
      %r = firrtl.reg %clock  : !firrtl.uint<1>
      %w = firrtl.wire  : !firrtl.bundle<a: uint<1>>
      %DataTap__0, %DataTap__1, %DataTap__2, %DataTap__3, %DataTap__4, %DataTap__5, %DataTap__6, %DataTap__7, %DataTap__8, %DataTap__9 = firrtl.instance @DataTap  {name = "DataTap"} : !firrtl.uint<1>, !firrtl.vector<uint<1>, 1>, !firrtl.uint<1>, !firrtl.vector<uint<1>, 1>, !firrtl.uint<1>, !firrtl.vector<uint<1>, 1>, !firrtl.uint<1>, !firrtl.vector<uint<1>, 1>, !firrtl.uint<1>, !firrtl.vector<uint<1>, 1>
      firrtl.instance @BlackBox  {name = "BlackBox"}
    }
  }

    // CHECK-LABEL: firrtl.circuit "GCTDataTap"
    // CHECK-SAME: annotations
    // CHECK-SAME: unrelatedAnnotation
    // CHECK: firrtl.extmodule @DataTap
    // CHECK-SAME: attributes {defname = "DataTap", firrtl.DoNotTouch = true, portAnnotations
    // CHECK-SAME:   class = "sifive.enterprise.grandcentral.ReferenceDataTapKey"
    // CHECK-SAME:    id = [[ID:[0-9]+]] : i64,
    // CHECK-SAME:    portID = [[PORT_ID_0:[0-9]+]] : i64,
    // CHECK-SAME:    type = "portName"
   // CHECK-SAME: circt.fieldID = 1 : i32,
   // CHECK-SAME: class = "sifive.enterprise.grandcentral.ReferenceDataTapKey"
   // CHECK-SAME: id = 0 : i64, portID = 2 : i64
   // CHECK-SAME: type = "portName"
   // CHECK-SAME: class = "sifive.enterprise.grandcentral.ReferenceDataTapKey"
   // CHECK-SAME: id = 0 : i64, portID = [[PORT_ID_3:[0-9]+]] : i64, type = "portName"
   // CHECK-SAME: circt.fieldID = 1 : i32
   // CHECK-SAME: class = "sifive.enterprise.grandcentral.ReferenceDataTapKey"
   // CHECK-SAME: id = 0 : i64, portID = [[PORT_ID_40:[0-9]+]] : i64, type = "portName"
   // CHECK-SAME: class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey"
   // CHECK-SAME: id = 0 : i64, portID = [[PORT_ID_4:[0-9]+]] : i64
   // CHECK-SAME: circt.fieldID = 1 : i32, class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey"
   // CHECK-SAME: id = 0 : i64, portID = [[PORT_ID_5:[0-9]+]] : i64
   // CHECK-SAME: class = "sifive.enterprise.grandcentral.DeletedDataTapKey"
   // CHECK-SAME: id = 0 : i64
   // CHECK-SAME: circt.fieldID = 1 : i32
   // CHECK-SAME: class = "sifive.enterprise.grandcentral.DeletedDataTapKey"
   // CHECK-SAME: id = 0 : i64
   // CHECK-SAME: class = "sifive.enterprise.grandcentral.LiteralDataTapKey"
   // CHECK-SAME: literal = "UInt<16>(\22h2a\22)"
   // CHECK-SAME: circt.fieldID = 1 : i32
   // CHECK-SAME: class = "sifive.enterprise.grandcentral.LiteralDataTapKey"
   // CHECK-SAME: literal = "UInt<16>(\22h2a\22)"

    // CHECK: firrtl.extmodule @BlackBox
    // CHECK-SAME: annotations
    // CHECK-SAME:   class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey"
    // CHECK-SAME:    id = [[ID]] : i64,
    // CHECK-SAME:    internalPath = "baz.qux",
    // CHECK-SAME:    portID = [[PORT_ID_4]] : i64
    // CHECK-SAME:   class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey"
    // CHECK-SAME:    id = [[ID]] : i64
    // CHECK-SAME:    internalPath = "baz.quz"
    // CHECK-SAME:    portID = [[PORT_ID_5]] : i64
    // CHECK-SAME:   firrtl.DoNotTouch = true

    // CHECK: firrtl.module @GCTDataTap
    // CHECK: firrtl.reg
    // CHECK-SAME: annotations =
    // CHECK-SAME:   class = "sifive.enterprise.grandcentral.ReferenceDataTapKey"
    // CHECK-SAME:    id = [[ID]] : i64
    // CHECK-SAME:    portID = [[PORT_ID_0]] : i64
    // CHECK-SAME:    type = "source"
    // CHECK-SAME:   firrtl.DoNotTouch = true

    // CHECK: firrtl.wire
    // CHECK-SAME: annotations =
    // CHECK-SAME: circt.fieldID = 1 : i32
    // CHECK-SAME: class = "sifive.enterprise.grandcentral.ReferenceDataTapKey"
    // CHECK-SAME: id = 0 : i64, portID = [[PORT_ID_3]] : i64, type = "source"
    // CHECK-SAME: circt.fieldID = 1 : i32
    // CHECK-SAME: class = "sifive.enterprise.grandcentral.ReferenceDataTapKey"
    // CHECK-SAME: id = 0 : i64, portID = [[PORT_ID_40]] : i64, type = "source"

// -----

  firrtl.circuit "GCTMemTap"  attributes {annotations = [{class = "sifive.enterprise.grandcentral.MemTapAnnotation", source = "~GCTMemTap|GCTMemTap>mem", taps = ["GCTMemTap.MemTap.mem[0]", "GCTMemTap.MemTap.mem[1]"]}, {class = "circt.testNT", unrelatedAnnotation}]}  {
    firrtl.extmodule @MemTap(out %mem: !firrtl.vector<uint<1>, 2>) attributes {defname = "MemTap"}
    firrtl.module @GCTMemTap(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
      %mem = firrtl.combmem  {name = "mem"} : !firrtl.cmemory<uint<1>, 2>
      %MemTap_mem = firrtl.instance @MemTap  {name = "MemTap"} : !firrtl.vector<uint<1>, 2>
      %0 = firrtl.subindex %MemTap_mem[0] : !firrtl.vector<uint<1>, 2>
      %1 = firrtl.subindex %MemTap_mem[1] : !firrtl.vector<uint<1>, 2>
      %memTap = firrtl.wire  : !firrtl.vector<uint<1>, 2>
      %2 = firrtl.subindex %memTap[0] : !firrtl.vector<uint<1>, 2>
      %3 = firrtl.subindex %MemTap_mem[0] : !firrtl.vector<uint<1>, 2>
      firrtl.connect %2, %3 : !firrtl.uint<1>, !firrtl.uint<1>
      %4 = firrtl.subindex %memTap[1] : !firrtl.vector<uint<1>, 2>
      %5 = firrtl.subindex %MemTap_mem[1] : !firrtl.vector<uint<1>, 2>
      firrtl.connect %4, %5 : !firrtl.uint<1>, !firrtl.uint<1>
    }
  }

// CHECK-LABEL: firrtl.circuit "GCTMemTap"
// CHECK-SAME: annotations
// CHECK-SAME: unrelatedAnnotation
// CHECK: firrtl.extmodule @MemTap
// CHECK-SAME: portAnnotations
// CHECK-SAME: circt.fieldID = 1 : i32
// CHECK-SAME: class = "sifive.enterprise.grandcentral.MemTapAnnotation"
// CHECK-SAME:      id = [[ID:[0-9]+]] : i64
// CHECK-SAME: circt.fieldID = 2 : i32
// CHECK-SAME: class = "sifive.enterprise.grandcentral.MemTapAnnotation"
// CHECK-SAME:      id = [[ID]] : i64
// CHECK: firrtl.module @GCTMemTap
// CHECK: %mem = firrtl.combmem
// CHECK-SAME: annotations = [
// CHECK-SAME:   class = "sifive.enterprise.grandcentral.MemTapAnnotation",
// CHECK-SAME:    id = [[ID]] : i64
// -----

  firrtl.circuit "GCTInterface"  attributes {annotations = [{class = "sifive.enterprise.grandcentral.GrandCentralView$SerializedViewAnnotation", companion = "~GCTInterface|view_companion", name = "view", parent = "~GCTInterface|GCTInterface", view = {class = "sifive.enterprise.grandcentral.AugmentedBundleType", defName = "ViewName", elements = [{description = "the register in GCTInterface", name = "register", tpe = {class = "sifive.enterprise.grandcentral.AugmentedBundleType", defName = "register", elements = [{name = "_2", tpe = {class = "sifive.enterprise.grandcentral.AugmentedVectorType", elements = [{class = "sifive.enterprise.grandcentral.AugmentedGroundType", ref = {circuit = "GCTInterface", component = [{class = "firrtl.annotations.TargetToken$Field", value = "_2"}, {class = "firrtl.annotations.TargetToken$Index", value = 0 : i64}], module = "GCTInterface", path = [], ref = "r"}, tpe = {class = "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"}}, {class = "sifive.enterprise.grandcentral.AugmentedGroundType", ref = {circuit = "GCTInterface", component = [{class = "firrtl.annotations.TargetToken$Field", value = "_2"}, {class = "firrtl.annotations.TargetToken$Index", value = 1 : i64}], module = "GCTInterface", path = [], ref = "r"}, tpe = {class = "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"}}]}}, {name = "_0", tpe = {class = "sifive.enterprise.grandcentral.AugmentedBundleType", defName = "_0", elements = [{name = "_1", tpe = {class = "sifive.enterprise.grandcentral.AugmentedGroundType", ref = {circuit = "GCTInterface", component = [{class = "firrtl.annotations.TargetToken$Field", value = "_0"}, {class = "firrtl.annotations.TargetToken$Field", value = "_1"}], module = "GCTInterface", path = [], ref = "r"}, tpe = {class = "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"}}}, {name = "_0", tpe = {class = "sifive.enterprise.grandcentral.AugmentedGroundType", ref = {circuit = "GCTInterface", component = [{class = "firrtl.annotations.TargetToken$Field", value = "_0"}, {class = "firrtl.annotations.TargetToken$Field", value = "_0"}], module = "GCTInterface", path = [], ref = "r"}, tpe = {class = "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"}}}]}}]}}, {description = "the port 'a' in GCTInterface", name = "port", tpe = {class = "sifive.enterprise.grandcentral.AugmentedGroundType", ref = {circuit = "GCTInterface", component = [], module = "GCTInterface", path = [], ref = "a"}, tpe = {class = "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"}}}]}}, {class = "circt.testNT", unrelatedAnnotation}]}  {
    firrtl.module @view_companion() {
      firrtl.skip
    }
    firrtl.module @GCTInterface(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %a: !firrtl.uint<1>) {
      %r = firrtl.reg %clock  : !firrtl.bundle<_0: bundle<_0: uint<1>, _1: uint<1>>, _2: vector<uint<1>, 2>>
      firrtl.instance @view_companion  {name = "view_companion"}
    }
  }
// CHECK-LABEL: firrtl.circuit "GCTInterface"
// The interface definitions should show up as circuit annotations.
// CHECK-SAME: annotations
// CHECK-SAME: unrelatedAnnotation
// CHECK-SAME: {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
// CHECK-SAME:   defName = "_0",
// CHECK-SAME:   elements = [
// CHECK-SAME:     {name = "_1",
// CHECK-SAME:      tpe = "sifive.enterprise.grandcentral.AugmentedGroundType"},
// CHECK-SAME:     {name = "_0",
// CHECK-SAME:      tpe = "sifive.enterprise.grandcentral.AugmentedGroundType"}]}
// CHECK-SAME: {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
// CHECK-SAME:   defName = "register",
// CHECK-SAME:   elements = [
// CHECK-SAME:     {name = "_2",
// CHECK-SAME:      tpe = "sifive.enterprise.grandcentral.AugmentedVectorType"},
// CHECK-SAME:     {name = "_0",
// CHECK-SAME:      tpe = "sifive.enterprise.grandcentral.AugmentedBundleType"}]}
// CHECK-SAME: {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
// CHECK-SAME:   defName = "ViewName",
// CHECK_SAME:   elements = [
// CHECK-SAME:     {description = "the register in GCTInterface",
// CHECK-SAME:      name = "register",
// CHECK-SAME:      tpe = "sifive.enterprise.grandcentral.AugmentedBundleType"},
// CHECK-SAME:     {description = "the port 'a' in GCTInterface",
// CHECK-SAME:      name = "port",
// CHECK-SAME:      tpe = "sifive.enterprise.grandcentral.AugmentedGroundType"}]}
//
// The companion should be marked.
// CHECK: firrtl.module @view_companion
// CHECK-SAME: annotations
// CHECK-SAME: class = "sifive.enterprise.grandcentral.ViewAnnotation",
// CHECK-SAME:  id = [[ID:.+]] : i64,
// CHECK-SAME:  type = "companion"}

// The parent should be annotated. Additionally, this example has all the
// members of the interface inside the parent.  Both port "a" and register
// "r" should be annotated.
// CHECK: firrtl.module @GCTInterface
// CHECK-SAME: annotations 
// CHECK-SAME:   class = "sifive.enterprise.grandcentral.ViewAnnotation"
// CHECK-SAME:   defName = "ViewName", id = [[ID]] : i64
// CHECK-SAME:   name = "view", type = "parent"
// CHECK-SAME: firrtl.DoNotTouch = true
// CHECK-SAME: class = "sifive.enterprise.grandcentral.AugmentedGroundType"
// CHECK-SAME:  defName = "ViewName", name = "port"
// CHECK: firrtl.reg
// CHECK-SAME: annotations
// CHECK-SAME: circt.fieldID = 5 : i32
// CHECK-SAME: class = "sifive.enterprise.grandcentral.AugmentedGroundType"
// CHECK-SAME: defName = "register", name = "_2"
// CHECK-SAME: circt.fieldID = 6 : i32
// CHECK-SAME: class = "sifive.enterprise.grandcentral.AugmentedGroundType"
// CHECK-SAME: defName = "register", name = "_2"
// CHECK-SAME: circt.fieldID = 3 : i32
// CHECK-SAME: class = "sifive.enterprise.grandcentral.AugmentedGroundType"
// CHECK-SAME: defName = "_0", name = "_1"
// CHECK-SAME: circt.fieldID = 2 : i32
// CHECK-SAME: class = "sifive.enterprise.grandcentral.AugmentedGroundType"
// CHECK-SAME: defName = "_0", name = "_0"
// CHECK-SAME: firrtl.DoNotTouch = true

// -----

  firrtl.circuit "Foo"  attributes {annotations = [{class = "sifive.enterprise.grandcentral.GrandCentralView$SerializedViewAnnotation", companion = "~Foo|Bar_companion", name = "Bar", parent = "~Foo|Foo", view = {class = "sifive.enterprise.grandcentral.AugmentedBundleType", defName = "View", elements = [{description = "a string", name = "string", tpe = {class = "sifive.enterprise.grandcentral.AugmentedStringType", value = "hello"}}, {description = "a boolean", name = "boolean", tpe = {class = "sifive.enterprise.grandcentral.AugmentedBooleanType", value = false}}, {description = "an integer", name = "integer", tpe = {class = "sifive.enterprise.grandcentral.AugmentedIntegerType", value = 42 : i64}}, {description = "a double", name = "double", tpe = {class = "sifive.enterprise.grandcentral.AugmentedDoubleType", value = 3.140000e+00 : f64}}]}}]}  {
    firrtl.extmodule @Bar_companion()
    firrtl.module @Foo() {
      firrtl.instance @Bar_companion  {name = "Bar_companion"}
    }
  }

// CHECK-LABEL:  firrtl.circuit "Foo"
// CHECK-SAME: class = "sifive.enterprise.grandcentral.AugmentedBundleType"
// CHECK-SAME: defName = "View"
// CHECK-SAME: elements = [{description = "a string", name = "string"
// CHECK-SAME: tpe = "sifive.enterprise.grandcentral.AugmentedStringType"
// CHECK-SAME: description = "a boolean", name = "boolean"
// CHECK-SAME: tpe = "sifive.enterprise.grandcentral.AugmentedBooleanType"
// CHECK-SAME: description = "an integer", name = "integer"
// CHECK-SAME: tpe = "sifive.enterprise.grandcentral.AugmentedIntegerType"
// CHECK-SAME: description = "a double", name = "double"
// CHECK-SAME: tpe = "sifive.enterprise.grandcentral.AugmentedDoubleType"

// CHECK-LABEL: firrtl.extmodule @Bar_companion()
// CHECK-SAME: class = "sifive.enterprise.grandcentral.ViewAnnotation", defName = "View", id = 9 : i64, type = "companion"
// CHECK-LABEL: firrtl.module @Foo()
// CHECK-SAME: class = "sifive.enterprise.grandcentral.ViewAnnotation", defName = "View", id = 9 : i64, name = "Bar", type = "parent"
