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
// CHECK: firrtl.wire {annotations = [{circt.nonlocal.key = "~FooNL|FooNL/baz:BazNL/bar:BarNL>w", circt.nonlocal.parent = "bar", class = "circt.test", nl = "nl"}]}
// CHECK: firrtl.instance @BarNL {annotations = [{circt.nonlocal.key = "~FooNL|FooNL/baz:BazNL/bar:BarNL>w", circt.nonlocal.parent = "baz", class = "circt.nonlocal"}], name = "bar"}
// CHECK: firrtl.instance @BazNL {annotations = [{circt.nonlocal.child = "bar", circt.nonlocal.key = "~FooNL|FooNL/baz:BazNL/bar:BarNL>w", class = "circt.nonlocal"}], name = "baz"
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

// A ReferenceTarget/ComponentName pointing at a CMem should work.

// CHECK-LABEL: firrtl.module @Foo7
// CHECK: firrtl.cmem
// CHECK-SAME: annotations = [{a = "a", class = "circt.test"}, {b = "b", class = "circt.test"}]
firrtl.circuit "Foo7"  attributes {annotations = [{a = "a", class = "circt.test", target = "~Foo7|Foo7>bar"}, {b = "b", class = "circt.test", target = "Foo7.Foo7.bar"}]}  {
  firrtl.module @Foo7() {
    %bar = firrtl.cmem  {name = "bar"} : !firrtl.vector<uint<1>, 8>
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

// CHECK-LABEL: firrtl.smem
// CHECK-SAME: annotations = [{a = "a", class = "circt.test"}, {b = "b", class = "circt.test"}]
firrtl.circuit "Foo"  attributes {annotations = [{a = "a", class = "circt.test", target = "~Foo|Foo>bar"}, {b = "b", class = "circt.test", target = "Foo.Foo.bar"}]}  {
  firrtl.module @Foo() {
    %bar = firrtl.smem Undefined  {name = "bar"} : !firrtl.vector<uint<1>, 8>
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

