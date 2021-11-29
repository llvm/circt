// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-lower-annotations)' -split-input-file %s |FileCheck %s

// circt.test copies the annotation to the target
// circt.testNT puts the targetless annotation on the circuit


// A non-local annotation should work.

// CHECK-LABEL: firrtl.circuit "FooNL"
// CHECK: firrtl.nla @nla_0 [@FooNL, @BazNL, @BarNL] ["baz", "bar", "w"]
// CHECK: firrtl.nla @nla [@FooNL, @BazNL, @BarNL] ["baz", "bar", "w2"] 
// CHECK: firrtl.module @BarNL
// CHECK: %w = firrtl.wire {annotations = [{circt.nonlocal = @nla_0, class = "circt.test", nl = "nl"}]}
// CHECK: %w2 = firrtl.wire {annotations = [{circt.fieldID = 5 : i32, circt.nonlocal = @nla, class = "circt.test", nl = "nl2"}]} : !firrtl.bundle<a: uint, b: vector<uint, 4>> 
// CHECK: firrtl.instance bar {annotations = [{circt.nonlocal = @nla, class = "circt.nonlocal"}, {circt.nonlocal = @nla_0, class = "circt.nonlocal"}]} @BarNL()
// CHECK: firrtl.instance baz {annotations = [{circt.nonlocal = @nla, class = "circt.nonlocal"}, {circt.nonlocal = @nla_0, class = "circt.nonlocal"}]} @BazNL()
// CHECK: firrtl.module @FooL
// CHECK: %w3 = firrtl.wire {annotations = [{class = "circt.test", nl = "nl3"}]}
firrtl.circuit "FooNL"  attributes {raw_annotations = [
  {class = "circt.test", nl = "nl", target = "~FooNL|FooNL/baz:BazNL/bar:BarNL>w"},
  {class = "circt.test", nl = "nl2", target = "~FooNL|FooNL/baz:BazNL/bar:BarNL>w2.b[2]"},
  {class = "circt.test", nl = "nl3", target = "~FooNL|FooL>w3"}
  ]}  {
  firrtl.module @BarNL() {
    %w = firrtl.wire  : !firrtl.uint
    %w2 = firrtl.wire  : !firrtl.bundle<a: uint, b: vector<uint, 4>>
    firrtl.skip
  }
  firrtl.module @BazNL() {
    firrtl.instance bar @BarNL()
  }
  firrtl.module @FooNL() {
    firrtl.instance baz @BazNL()
  }
  firrtl.module @FooL() {
    %w3 = firrtl.wire: !firrtl.uint
  }
}


// -----

firrtl.circuit "Foo"  attributes {raw_annotations = [{a = "a", class = "circt.test", target = "~"}]}  {
  firrtl.module @Foo() {
    firrtl.skip
  }
}

// -----
firrtl.circuit "Foo"  attributes {raw_annotations = [{a = "a", target = "~Foo"}]}  {
  firrtl.module @Foo() {
    firrtl.skip
  }
}

// -----
firrtl.circuit "Foo"  attributes {raw_annotations = [{a = "a", target = "~Foo"}]}  {
  firrtl.module @Foo() {
    firrtl.skip
  }
}

// -----
firrtl.circuit "Foo"  attributes {raw_annotations = [{a = "a", target = "~Foo|Foo"}]}  {
  firrtl.module @Foo() {
    firrtl.skip
  }
}

// -----

firrtl.circuit "Foo"  attributes {raw_annotations = [{a = "a", target = "~Foo|Foo"}]}  {
  firrtl.module @Foo() {
    firrtl.skip
  }
}

// -----

// A ModuleTarget Annotation is attached to the correct module.

firrtl.circuit "Foo"  attributes {raw_annotations = [{a = "a", target = "~Foo|Bar"}]}  {
  firrtl.extmodule @Bar(in a: !firrtl.uint<1>)
  firrtl.module @Foo(in %a: !firrtl.uint<1>) {
    %bar_a = firrtl.instance bar  @Bar(in a: !firrtl.uint<1>)
    firrtl.connect %bar_a, %a : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

    // CHECK-LABEL: firrtl.circuit "Foo" {
    // CHECK: firrtl.module @Foo() attributes {raw_annotations = [{a = "a", target = "~Foo|Foo"}]}

// -----

// A ReferenceTarget, ComponentName, or InstanceTarget pointing at an Instance
// should work.

firrtl.circuit "Foo"  attributes {raw_annotations = [{a = "a", target = "~Foo|Foo>bar"}, {b = "b", target = "~Foo|Foo>bar"}, {c = "c", target = "~Foo|Foo/bar:Bar"}]}  {
  firrtl.module @Bar() {
    firrtl.skip
  }
  firrtl.module @Foo() {
    firrtl.instance bar  @Bar()
  }
}
    // CHECK-LABEL: firrtl.circuit "Foo"
    // CHECK: firrtl.nla  @nla_1 [@Foo, @Bar] ["bar", "Bar"]
    // CHECK: firrtl.module @Bar
    // CHECK-SAME annotations = [{c = "c"}]
    // CHECK: firrtl.module @Foo
    // CHECK: firrtl.instance bar
    // CHECK-SAME: annotations = [{a = "a"}, {b = "b"}, {circt.nonlocal = @nla_1, class = "circt.nonlocal"}]

// -----

// Test result annotations of InstanceOp.

firrtl.circuit "Foo"  attributes {raw_annotations = [{one, target = "~Foo|Foo>bar.a"}, {target = "~Foo|Foo>bar.b.baz", two}, {target = "~Foo|Foo/bar:Bar>b.qux", three}, {four, target = "~Foo|Foo>bar'c"}]}  {
  firrtl.module @Bar(in %a: !firrtl.uint<1>, out %b: !firrtl.bundle<baz: uint<1>, qux: uint<1>>, out %c: !firrtl.uint<1>) {
  }
  firrtl.module @Foo() {
    %bar_a, %bar_b, %bar_c = firrtl.instance bar  @Bar(in a: !firrtl.uint<1>, out b: !firrtl.bundle<baz: uint<1>, qux: uint<1>>, out c: !firrtl.uint<1>)
  }
}

    // CHECK-LABEL: firrtl.circuit "Foo"
    // CHECK: firrtl.nla @nla_1 [@Foo, @Bar] ["bar", "b"]
    // CHECK: firrtl.module @Bar
    // CHECK-SAME: [#firrtl.subAnno<fieldID = 2, {circt.nonlocal = @nla_1, three}>]
    // CHECK: %bar_a, %bar_b, %bar_c = firrtl.instance bar
    // CHECK-SAME: [{one}],
    // CHECK-SAME: [#firrtl.subAnno<fieldID = 1, {two}>],
    // CHECK-SAME: [{four}]

// -----

// A ReferenceTarget/ComponentName pointing at a CombMem should work.

firrtl.circuit "Foo"  attributes {raw_annotations = [{a = "a", target = "~Foo|Foo>bar"}, {b = "b", target = "~Foo|Foo>bar"}]}  {
  firrtl.module @Foo() {
    %bar = firrtl.combmem  : !firrtl.cmemory<uint<1>, 8>
  }
}

    // CHECK-LABEL: module {
    // CHECK: firrtl.combmem
    // CHECK-SAME: annotations = [{a = "a"}, {b = "b"}]


// -----

 // A ReferenceTarget/ComponentName pointing at a memory should work.

firrtl.circuit "Foo"  attributes {raw_annotations = [{a = "a", target = "~Foo|Foo>bar"}, {b = "b", target = "~Foo|Foo>bar"}]}  {
  firrtl.module @Foo() {
    %bar_r, %bar_w = firrtl.mem Undefined  {depth = 16 : i64, name = "bar", portNames = ["r", "w"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
  }
}
    // CHECK-LABEL: module {
    // CHECK: firrtl.mem
    // CHECK-SAME: annotations = [{a = "a"}, {b = "b"}]

// -----
 
// Test result annotations of MemOp.

firrtl.circuit "Foo"  attributes {raw_annotations = [{a, target = "~Foo|Foo>bar.r"}, {b, target = "~Foo|Foo>bar.r.data.baz"}, {c, target = "~Foo|Foo>bar.w.en"}, {d, target = "~Foo|Foo>bar.w.data.qux"}]}  {
  firrtl.module @Foo() {
    %bar_r, %bar_w = firrtl.mem Undefined  {depth = 16 : i64, name = "bar", portNames = ["r", "w"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: bundle<baz: uint<8>, qux: uint<8>>>, !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data: bundle<baz: uint<8>, qux: uint<8>>, mask: bundle<baz: uint<1>, qux: uint<1>>>
  }
}

    // CHECK-LABEL: module {
    // CHECK: firrtl.mem
    // CHECK-SAME: portAnnotations = [
    // CHECK-SAME: [{a}, #firrtl.subAnno<fieldID = 5, {b}>],
    // CHECK-SAME: [#firrtl.subAnno<fieldID = 2, {c}>, #firrtl.subAnno<fieldID = 6, {d}>]


// -----
 
// A ReferenceTarget/ComponentName pointing at a node should work.  This
// shouldn't crash if the node is in a nested block.

firrtl.circuit "Foo"  attributes {raw_annotations = [{a = "a", target = "~Foo|Foo>bar"}, {b = "b", target = "~Foo|Foo>baz"}]}  {
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

    // CHECK-LABEL: module {
    // CHECK: %bar = firrtl.node
    // CHECK-SAME: annotations = [{a = "a"}
    // CHECK: %baz = firrtl.node
    // CHECK-SAME: annotations = [{b = "b"}]

// -----

// A ReferenceTarget/ComponentName pointing at a wire should work.

firrtl.circuit "Foo"  attributes {raw_annotations = [{a = "a", target = "~Foo|Foo>bar"}, {b = "b", target = "~Foo|Foo>bar"}]}  {
  firrtl.module @Foo() {
    %bar = firrtl.wire  : !firrtl.uint<1>
  }
}
 
   // CHECK-LABEL: module {
    // CHECK: %bar = firrtl.wire
    // CHECK-SAME: annotations = [{a = "a"}, {b = "b"}]

// -----
 
 // A ReferenceTarget/ComponentName pointing at a register should work.

firrtl.circuit "Foo"  attributes {raw_annotations = [{a = "a", target = "~Foo|Foo>bar"}, {b = "b", target = "~Foo|Foo>baz"}]}  {
  firrtl.module @Foo(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
    %bar = firrtl.reg %clock  : !firrtl.uint<1>
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %baz = firrtl.regreset %clock, %reset, %c0_ui1  : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
  }
}

    // CHECK-LABEL: module {
    // CHECK: %bar = firrtl.reg
    // CHECK-SAME: annotations = [{a = "a"}]
    // CHECK: %baz = firrtl.regreset
    // CHECK-SAME: annotations = [{b = "b"}]

// -----
 
 // A ReferenceTarget/ComponentName pointing at an SeqMem should work.

firrtl.circuit "Foo"  attributes {raw_annotations = [{a = "a", target = "~Foo|Foo>bar"}, {b = "b", target = "~Foo|Foo>bar"}]}  {
  firrtl.module @Foo() {
    %bar = firrtl.seqmem Undefined  : !firrtl.cmemory<uint<1>, 8>
  }
}
    // CHECK-LABEL: module {
    // CHECK: firrtl.seqmem
    // CHECK-SAME: annotations = [{a = "a"}, {b = "b"}]

// -----

// A ReferenceTarget/ComponentName pointing at a module/extmodule port should work.

firrtl.circuit "Foo"  attributes {raw_annotations = [{a = "a", target = "~Foo|Bar>bar"}, {b = "b", target = "~Foo|Foo>foo"}]}  {
  firrtl.extmodule @Bar(in bar: !firrtl.uint<1>)
  firrtl.module @Foo(in %foo: !firrtl.uint<1>) {
    %bar_bar = firrtl.instance bar  @Bar(in bar: !firrtl.uint<1>)
    firrtl.connect %bar_bar, %foo : !firrtl.uint<1>, !firrtl.uint<1>
  }
}
    // CHECK-LABEL: module {
    // CHECK: firrtl.extmodule @Bar
    // CHECK-SAME: [[_:.+]] [{a = "a"}]
    // CHECK: firrtl.module @Foo
    // CHECK-SAME: %foo: [[_:.+]] [{b = "b"}]

// -----

 // All types of JSON values should work

firrtl.circuit "Foo"  attributes {raw_annotations = [{array = [1, 2, 3], boolean = true, float = 3.140000e+00 : f64, integer = 42 : i64, null, object = {foo = "bar"}, string = "a", target = "~"}]}  {
  firrtl.module @Foo() {
    firrtl.skip
  }
}
    // CHECK-LABEL: module {
    // CHECK: firrtl.circuit "Foo" attributes {annotations =
    // CHECK-SAME: array = [1, 2, 3]
    // CHECK-SAME: boolean = true
    // CHECK-SAME: float = 3.140
    // CHECK-SAME: integer = 42
    // CHECK-SAME: object = {foo = "bar"}
    // CHECK-SAME: string = "a"

// -----
 // JSON escapes should work.

firrtl.circuit "Foo"  attributes {raw_annotations = [{"\22" = "}]]", target = "~"}]}  {
  firrtl.module @Foo() {
    firrtl.skip
  }
}

    // CHECK-LABEL: module {
    // CHECK: firrtl.circuit "Foo" attributes {annotations =

// -----

// JSON with a JSON-quoted string should be expanded.

firrtl.circuit "Foo"  attributes {raw_annotations = [{a = {b}, target = "~"}]}  {
  firrtl.module @Foo() {
    firrtl.skip
  }
}

    // CHECK-LABEL: module {
    // CHECK: firrtl.circuit "Foo" attributes {annotations = [{a = {b}}]}

// -----
// Subfield/Subindex annotations should be parsed correctly on wires

firrtl.circuit "Foo"  attributes {raw_annotations = [{one, target = "~Foo|Foo>bar[0]"}, {target = "~Foo|Foo>bar[1].baz", two}]}  {
  firrtl.module @Foo() {
    %bar = firrtl.wire  : !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>
  }
}
    // CHECK-LABEL: module {
    // CHECK: %bar = firrtl.wire  {annotations =
    // CHECK-SAME: #firrtl.subAnno<fieldID = 1, {one}>
    // CHECK-SAME: #firrtl.subAnno<fieldID = 5, {two}>

// -----
// Subfield/Subindex annotations should be parsed correctly on registers

firrtl.circuit "Foo"  attributes {raw_annotations = [{one, target = "~Foo|Foo>bar[0]"}, {target = "~Foo|Foo>bar[1].baz", two}]}  {
  firrtl.module @Foo(in %clock: !firrtl.clock) {
    %bar = firrtl.reg %clock  : !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>
  }
}

    // CHECK-LABEL: module {
    // CHECK: %bar = firrtl.reg %clock  {annotations =
    // CHECK-SAME: #firrtl.subAnno<fieldID = 1, {one}>
    // CHECK-SAME: #firrtl.subAnno<fieldID = 5, {two}>

// -----
// Subindices should not get sign-extended and cause problems.  This circuit has
// caused bugs in the past.

firrtl.circuit "Foo"  attributes {raw_annotations = [{a, target = "~Foo|Foo>w[9]"}]}  {
  firrtl.module @Foo(in %a: !firrtl.vector<uint<1>, 18>, out %b: !firrtl.vector<uint<1>, 18>) {
    %w = firrtl.wire  : !firrtl.vector<uint<1>, 18>
    firrtl.connect %w, %a : !firrtl.vector<uint<1>, 18>, !firrtl.vector<uint<1>, 18>
    firrtl.connect %b, %w : !firrtl.vector<uint<1>, 18>, !firrtl.vector<uint<1>, 18>
  }
}

    // CHECK-LABEL: module {
    // CHECK: %w = firrtl.wire {annotations =
    // CHECK-SAME: #firrtl.subAnno<fieldID = 10, {a}

// -----
// Annotations should apply even when the target's name is dropped.

firrtl.circuit "Foo"  attributes {raw_annotations = [{a = "a", target = "~Foo|Foo>_T_0"}, {a = "a", target = "~Foo|Foo>_T_1"}, {a =
 "a", target = "~Foo|Foo>_T_2"}, {a = "a", target = "~Foo|Foo>_T_3"}, {a = "a", target = "~Foo|Foo>_T_4"}, {a = "a", target = "~Foo|Foo>_T_5"}, {a = "a", target = "~Foo|Foo>_T_6"}, {a = "a", target = "~Foo|Foo>_T_7"}, {a = "a", target = "~Foo|Foo>_T_8"}]}  {
  firrtl.module @Bar() {
    firrtl.skip
  }
  firrtl.module @Foo(in %reset: !firrtl.uint<1>, in %clock: !firrtl.clock) {
    // CHECK: %_T_0 = firrtl.wire  {annotations = [{a = "a"}]}
    %_T_0 = firrtl.wire  : !firrtl.uint<1>
    // CHECK: %_T_1 = firrtl.node
    %_T_1 = firrtl.node %_T_0  : !firrtl.uint<1>
    // CHECK: %_T_2 = firrtl.reg %clock  {annotations = [{a = "a"}]}
    %_T_2 = firrtl.reg %clock  : !firrtl.uint<1>
    // CHECK: %_T_3 = firrtl.regreset {{.+}}  {annotations = [{a = "a"}]}
    %c0_ui4 = firrtl.constant 0 : !firrtl.uint<4>
    %_T_3 = firrtl.regreset %clock, %reset, %c0_ui4  : !firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>
    %_T_4 = firrtl.seqmem Undefined  : !firrtl.cmemory<vector<uint<1>, 9>, 256>
    // CHECK: %_T_4 = firrtl.seqmem Undefined {annotations = [{a = "a"}]}
    %_T_5 = firrtl.combmem  : !firrtl.cmemory<vector<uint<1>, 9>, 256>
    // CHECK: %_T_5 = firrtl.combmem  {annotations = [{a = "a"}]}
    %_T_6_data, %_T_6_port = firrtl.memoryport Infer %_T_5  {name = "_T_6"} : (!firrtl.cmemory<vector<uint<1>, 9>, 256>) -> (!firrtl.vector<uint<1>, 9>, !firrtl.cmemoryport)
    firrtl.memoryport.access %_T_6_port[%reset], %clock : !firrtl.cmemoryport, !firrtl.uint<1>, !firrtl.clock
    // CHECK: firrtl.memoryport {{.+}} {annotations = [{a = "a"}]
    // CHECK: firrtl.instance _T_7 {annotations = [{a = "a"}]}
    firrtl.instance _T_7  @Bar()
    // CHECK: firrtl.mem Undefined  {annotations = [{a = "a"}]
    %_T_8_w = firrtl.mem Undefined  {depth = 8 : i64, name = "_T_8", portNames = ["w"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<4>, mask: uint<1>>
  }
}

// -----

 // DontTouch annotation preserves temporary names

firrtl.circuit "Foo"  attributes {raw_annotations = [{class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_0"}, {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_1"}, {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_2"}, {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_3"}, {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_4"}, {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_5"}, {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_6"}, {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_7"}, {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_8"}, {class = "firrtl.transforms.DontTouchAnnotation", target = "~Foo|Foo>_T_9.a"}]}  {
  firrtl.module @Bar() {
    firrtl.skip
  }
  firrtl.module @Foo(in %reset: !firrtl.uint<1>, in %clock: !firrtl.clock) {
    // CHECK: %_T_0 = firrtl.wire  {annotations =
    // CHECK-SAME: {class = "firrtl.transforms.DontTouchAnnotation"}
    %_T_0 = firrtl.wire  : !firrtl.uint<1>
    // CHECK: %_T_1 = firrtl.node %_T_0  {annotations =
    // CHECK-SAME: {class = "firrtl.transforms.DontTouchAnnotation"}
    %_T_1 = firrtl.node %_T_0  : !firrtl.uint<1>
    // CHECK: %_T_2 = firrtl.reg %clock  {annotations =
    // CHECK-SAME: {class = "firrtl.transforms.DontTouchAnnotation"}
    %_T_2 = firrtl.reg %clock  : !firrtl.uint<1>
    // CHECK: %_T_3 = firrtl.regreset %clock, %reset, %c0_ui4  {annotations =
    // CHECK-SAME: {class = "firrtl.transforms.DontTouchAnnotation"}
    %c0_ui4 = firrtl.constant 0 : !firrtl.uint<4>
    %_T_3 = firrtl.regreset %clock, %reset, %c0_ui4  : !firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>
    // CHECK: %_T_4 = firrtl.seqmem Undefined  {annotations =
    // CHECK-SAME: {class = "firrtl.transforms.DontTouchAnnotation"}
    %_T_4 = firrtl.seqmem Undefined  : !firrtl.cmemory<vector<uint<1>, 9>, 256>
    // CHECK: %_T_5 = firrtl.combmem  {annotations = [
    // CHECK-SAME: {class = "firrtl.transforms.DontTouchAnnotation"}
    %_T_5 = firrtl.combmem  : !firrtl.cmemory<vector<uint<1>, 9>, 256>
    // CHECK: firrtl.memoryport Infer %_T_5 {annotations =
    // CHECK-SAME: {class = "firrtl.transforms.DontTouchAnnotation"}
    %_T_6_data, %_T_6_port = firrtl.memoryport Infer %_T_5  {name = "_T_6"} : (!firrtl.cmemory<vector<uint<1>, 9>, 256>) -> (!firrtl.vector<uint<1>, 9>, !firrtl.cmemoryport)
    // CHECK: firrtl.instance _T_7 {annotations =
    // CHECK-SAME: {class = "firrtl.transforms.DontTouchAnnotation"}
    firrtl.memoryport.access %_T_6_port[%reset], %clock : !firrtl.cmemoryport, !firrtl.uint<1>, !firrtl.clock
    firrtl.instance _T_7  @Bar()
    // CHECK: firrtl.mem Undefined  {annotations =
    // CHECK_SAME: {class = "firrtl.transforms.DontTouchAnnotation"}
    %_T_8_w = firrtl.mem Undefined  {depth = 8 : i64, name = "_T_8", portNames = ["w"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<4>, mask: uint<1>>
    %aggregate = firrtl.wire  : !firrtl.bundle<a: uint<1>>
    // CHECK: %_T_9 = firrtl.node
    // CHECK-SAME: {class = "firrtl.transforms.DontTouchAnnotation"}
    %_T_9 = firrtl.node %aggregate  : !firrtl.bundle<a: uint<1>>
  }
}
 
// -----
 
// Test that an annotated, anonymous node is preserved if annotated.  Normally,
// the FIRRTL parser will aggressively eliminate these.

firrtl.circuit "AnnotationsBlockNodePruning"  attributes {raw_annotations = [{a, target = "~AnnotationsBlockNodePruning|AnnotationsBlockNodePruning>_T"}]}  {
  firrtl.module @AnnotationsBlockNodePruning(in %a: !firrtl.uint<1>) {
    %0 = firrtl.not %a : (!firrtl.uint<1>) -> !firrtl.uint<1>
    %_T = firrtl.node %0  : !firrtl.uint<1>
  }
}

    // CHECK-LABEL: firrtl.module @AnnotationsBlockNodePruning
    // CHECK: firrtl.node

// -----

// --------------------------------------------------------------------------------
// SiFive-custom annotations related to the GrandCentral utility.  These
// annotations do not conform to standard SingleTarget or NoTarget format and
// need to be manually split up.
// --------------------------------------------------------------------------------

// Test sifive.enterprise.grandcentral.DataTapsAnnotation with all possible
// variants of DataTapKeys.

firrtl.circuit "GCTDataTap"  attributes {raw_annotations = [{blackBox = "~GCTDataTap|DataTap", class = "sifive.enterprise.grandcentral.DataTapsAnnotation", keys = [{class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", portName = "~GCTDataTap|DataTap>_0", source = "~GCTDataTap|GCTDataTap>r"}, {class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", portName = "~GCTDataTap|DataTap>_1[0]", source = "~GCTDataTap|GCTDataTap>r"}, {class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", portName = "~GCTDataTap|DataTap>_2", source = "~GCTDataTap|GCTDataTap>w.a"}, {class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", portName = "~GCTDataTap|DataTap>_3[0]", source = "~GCTDataTap|GCTDataTap>w.a"}, {class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey", internalPath = "baz.qux", module = "~GCTDataTap|BlackBox", portName = "~GCTDataTap|DataTap>_4"}, {class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey", internalPath = "baz.quz", module = "~GCTDataTap|BlackBox", portName = "~GCTDataTap|DataTap>_5[0]"}, {class = "sifive.enterprise.grandcentral.DeletedDataTapKey", portName = "~GCTDataTap|DataTap>_6"}, {class = "sifive.enterprise.grandcentral.DeletedDataTapKey", portName = "~GCTDataTap|DataTap>_7[0]"}, {class = "sifive.enterprise.grandcentral.LiteralDataTapKey", literal = "UInt<16>(\22h2a\22)", portName = "~GCTDataTap|DataTap>_8"}, {class = "sifive.enterprise.grandcentral.LiteralDataTapKey", literal = "UInt<16>(\22h2a\22)", portName = "~GCTDataTap|DataTap>_9[0]"}, {class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", portName = "~GCTDataTap|DataTap>_10", source = "~GCTDataTap|GCTDataTap/im:InnerMod>w"}], target = "~"}, {target = "~", unrelatedAnnotation}]}  {
  firrtl.extmodule @DataTap(out _0: !firrtl.uint<1>, out _1: !firrtl.vector<uint<1>, 1>, out _2: !firrtl.uint<1>, out _3: !firrtl.vector<uint<1>, 1>, out _4: !firrtl.uint<1>, out _5: !firrtl.vector<uint<1>, 1>, out _6: !firrtl.uint<1>, out _7: !firrtl.vector<uint<1>, 1>, out _8: !firrtl.uint<1>, out _9: !firrtl.vector<uint<1>, 1>, out _10: !firrtl.uint<1>) attributes {defname = "DataTap"}
  firrtl.extmodule @BlackBox() attributes {defname = "BlackBox"}
  firrtl.module @InnerMod() {
    %w = firrtl.wire  : !firrtl.uint<1>
  }
  firrtl.module @GCTDataTap(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    %r = firrtl.reg %clock  : !firrtl.uint<1>
    %w = firrtl.wire  : !firrtl.bundle<a: uint<1>>
    %DataTap__0, %DataTap__1, %DataTap__2, %DataTap__3, %DataTap__4, %DataTap__5, %DataTap__6, %DataTap__7, %DataTap__8, %DataTap__9, %DataTap__10 = firrtl.instance DataTap  @DataTap(out _0: !firrtl.uint<1>, out _1: !firrtl.vector<uint<1>, 1>, out _2: !firrtl.uint<1>, out _3: !firrtl.vector<uint<1>, 1>, out _4: !firrtl.uint<1>, out _5: !firrtl.vector<uint<1>, 1>, out _6: !firrtl.uint<1>, out _7: !firrtl.vector<uint<1>, 1>, out _8: !firrtl.uint<1>, out _9: !firrtl.vector<uint<1>, 1>, out _10: !firrtl.uint<1>)
    firrtl.instance BlackBox  @BlackBox()
    firrtl.instance im  @InnerMod()
  }
}


    // CHECK-LABEL: firrtl.circuit "GCTDataTap"
    // CHECK-SAME: annotations = [{unrelatedAnnotation}]
    // CHECK:      firrtl.nla @nla_1 [@GCTDataTap, @InnerMod] ["im", "w"]
    // CHECK: firrtl.extmodule @DataTap
    // CHECK-SAME: _0: !firrtl.uint<1> [
    // CHECK-SAME:   {class = "firrtl.transforms.DontTouchAnnotation"}
    // CHECK-SAME:   {class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
    // CHECK-SAME:    id = [[ID:[0-9]+]] : i64,
    // CHECK-SAME:    portID = [[PORT_ID_0:[0-9]+]] : i64,
    // CHECK-SAME:    type = "portName"}
    // CHECK-SAME: _1: !firrtl.vector<uint<1>, 1> [
    // CHECK-SAME:   #firrtl.subAnno<fieldID = 1,
    // CHECK-SAME:     {class = "firrtl.transforms.DontTouchAnnotation"}>
    // CHECK-SAME:   #firrtl.subAnno<fieldID = 1,
    // CHECK-SAME:     {class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
    // CHECK-SAME:      id = [[ID]] : i64,
    // CHECK-SAME:      portID = [[PORT_ID_1:[0-9]+]] : i64,
    // CHECK-SAME:      type = "portName"}>
    // CHECK-SAME: _2: !firrtl.uint<1> [
    // CHECK-SAME:   {class = "firrtl.transforms.DontTouchAnnotation"}
    // CHECK-SAME:   {class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
    // CHECK-SAME:    id = [[ID]] : i64,
    // CHECK-SAME:    portID = [[PORT_ID_2:[0-9]+]] : i64,
    // CHECK-SAME:    type = "portName"}
    // CHECK-SAME: _3: !firrtl.vector<uint<1>, 1> [
    // CHECK-SAME:   #firrtl.subAnno<fieldID = 1,
    // CHECK-SAME:     {class = "firrtl.transforms.DontTouchAnnotation"}>
    // CHECK-SAME:   #firrtl.subAnno<fieldID = 1,
    // CHECK-SAME:     {class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
    // CHECK-SAME:      id = [[ID]] : i64,
    // CHECK-SAME:      portID = [[PORT_ID_3:[0-9]+]] : i64,
    // CHECK-SAME:      type = "portName"}>
    // CHECK-SAME: _4: !firrtl.uint<1> [
    // CHECK-SAME:   {class = "firrtl.transforms.DontTouchAnnotation"}
    // CHECK-SAME:   {class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey",
    // CHECK-SAME:    id = [[ID]] : i64,
    // CHECK-SAME:    portID = [[PORT_ID_4:[0-9]+]] : i64}
    // CHECK-SAME: _5: !firrtl.vector<uint<1>, 1> [
    // CHECK-SAME:   #firrtl.subAnno<fieldID = 1,
    // CHECK-SAME:     {class = "firrtl.transforms.DontTouchAnnotation"}>
    // CHECK-SAME:   #firrtl.subAnno<fieldID = 1,
    // CHECK-SAME:     {class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey",
    // CHECK-SAME:      id = [[ID]] : i64,
    // CHECK-SAME:      portID = [[PORT_ID_5:[0-9]+]] : i64}>
    // CHECK-SAME: _6: !firrtl.uint<1> [
    // CHECK-SAME:   {class = "firrtl.transforms.DontTouchAnnotation"}
    // CHECK-SAME:   {class = "sifive.enterprise.grandcentral.DeletedDataTapKey",
    // CHECK-SAME:    id = [[ID]] : i64}
    // CHECK-SAME: _7: !firrtl.vector<uint<1>, 1> [
    // CHECK-SAME:   #firrtl.subAnno<fieldID = 1,
    // CHECK-SAME:     {class = "firrtl.transforms.DontTouchAnnotation"}>
    // CHECK-SAME:   #firrtl.subAnno<fieldID = 1,
    // CHECK-SAME:     {class = "sifive.enterprise.grandcentral.DeletedDataTapKey",
    // CHECK-SAME:      id = [[ID]] : i64}>
    // CHECK-SAME: _8: !firrtl.uint<1> [
    // CHECK-SAME:   {class = "firrtl.transforms.DontTouchAnnotation"}
    // CHECK-SAME:   {class = "sifive.enterprise.grandcentral.LiteralDataTapKey",
    // CHECK-SAME:    literal = "UInt<16>(\22h2a\22)"}
    // CHECK-SAME: _9: !firrtl.vector<uint<1>, 1> [
    // CHECK-SAME:   #firrtl.subAnno<fieldID = 1,
    // CHECK-SAME:     {class = "firrtl.transforms.DontTouchAnnotation"}>
    // CHECK-SAME:     {class = "sifive.enterprise.grandcentral.LiteralDataTapKey",
    // CHECK-SAME:      literal = "UInt<16>(\22h2a\22)"}
    // CHECK-SAME: _10: !firrtl.uint<1> [
    // CHECK-SAME      {class = "firrtl.transforms.DontTouchAnnotation"}
    // CHECK-SAME:     {class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = [[ID]] : i64, portID = [[PORT_ID_6:[0-9]+]] : i64, type = "portName"}
    // CHECK-SAME: annotations = [
    // CHECK-SAME:   {class = "sifive.enterprise.grandcentral.DataTapsAnnotation"},
    // CHECK-SAME:   {class = "firrtl.transforms.DontTouchAnnotation"}]

    // CHECK: firrtl.extmodule @BlackBox
    // CHECK-SAME: annotations = [
    // CHECK-SAME:   {class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey",
    // CHECK-SAME:    id = [[ID]] : i64,
    // CHECK-SAME:    internalPath = "baz.qux",
    // CHECK-SAME:    portID = [[PORT_ID_4]] : i64}
    // CHECK-SAME:   {class = "firrtl.transforms.DontTouchAnnotation"}
    // CHECK-SAME:   {class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey",
    // CHECK-SAME:    id = [[ID]] : i64,
    // CHECK-SAME:    internalPath = "baz.quz",
    // CHECK-SAME:    portID = [[PORT_ID_5]] : i64}

    // CHECK: firrtl.module @InnerMod
    // CHECK-NEXT: %w = firrtl.wire
    // CHECK-SAME: {circt.nonlocal = @nla_1, class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = 0 : i64, portID = 7 : i64, type = "source"}
    // CHECK-SAME: {class = "firrtl.transforms.DontTouchAnnotation"}

    // CHECK: firrtl.module @GCTDataTap
    // CHECK: firrtl.reg
    // CHECk-SAME: annotations =
    // CHECK-SAME:   {class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
    // CHECK-SAME:    id = [[ID]] : i64,
    // CHECK-SAME:    portID = [[PORT_ID_0]] : i64,
    // CHECK-SAME:    type = "source"}
    // CHECK-SAME:   {class = "firrtl.transforms.DontTouchAnnotation"}

    // CHECK: firrtl.wire
    // CHECK-SAME: annotations =
    // CHECK-SAME:   #firrtl.subAnno<fieldID = 1,
    // CHECK-SAME:     {class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
    // CHECK-SAME:      id = [[ID]] : i64,
    // CHECK-SAME:      portID = [[PORT_ID_2]] : i64,
    // CHECK-SAME:      type = "source"}>
    // CHECK-SAME:   #firrtl.subAnno<fieldID = 1,
    // CHECK-SAME:     {class = "firrtl.transforms.DontTouchAnnotation"}>
    // CHECK-SAME:   #firrtl.subAnno<fieldID = 1,
    // CHECK-SAME:     {class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
    // CHECK-SAME:      id = [[ID]] : i64,
    // CHECK-SAME:      portID = [[PORT_ID_3]] : i64,
    // CHECK-SAME:      type = "source"}>

// -----

// Test sifive.enterprise.grandcentral.MemTapAnnotation

firrtl.circuit "GCTMemTap"  attributes {raw_annotations = [{class = "sifive.enterprise.grandcentral.MemTapAnnotation", source = "~GCTMemTap|GCTMemTap>mem", taps = ["GCTMemTap.MemTap.mem[0]", "GCTMemTap.MemTap.mem[1]"], target = "~"}, {target = "~", unrelatedAnnotation}]}  {
  firrtl.extmodule @MemTap(out mem: !firrtl.vector<uint<1>, 2>) attributes {defname = "MemTap"}
  firrtl.module @GCTMemTap(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
    %mem = firrtl.combmem  : !firrtl.cmemory<uint<1>, 2>
    %MemTap_mem = firrtl.instance MemTap  @MemTap(out mem: !firrtl.vector<uint<1>, 2>)
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
    // CHECK-SAME: annotations = [{unrelatedAnnotation}]
    // CHECK: firrtl.extmodule @MemTap
    // CHECK-SAME: mem: [[A:.+]] [
    // CHECK-SAME:   #firrtl.subAnno<fieldID = 1,
    // CHECK-SAME:     {class = "sifive.enterprise.grandcentral.MemTapAnnotation",
    // CHECK-SAME:      id = [[ID:[0-9]+]] : i64, word = 0 : i64}>
    // CHECK-SAME:   #firrtl.subAnno<fieldID = 2,
    // CHECK-SAME:     {class = "sifive.enterprise.grandcentral.MemTapAnnotation",
    // CHECK-SAME:      id = [[ID]] : i64, word = 1 : i64}>
    // CHECK: firrtl.module @GCTMemTap
    // CHECK: %mem = firrtl.combmem
    // CHECK-SAME: annotations = [
    // CHECK-SAME:   {class = "sifive.enterprise.grandcentral.MemTapAnnotation",
    // CHECK-SAME:    id = [[ID]] : i64}]


// -----

// Test sifive.enterprise.grandcentral.ViewAnnotation

firrtl.circuit "GCTInterface"  attributes {raw_annotations = [{class = "sifive.enterprise.grandcentral.GrandCentralView$SerializedViewAnnotation", companion = "~GCTInterface|view_companion", name = "view", parent = "~GCTInterface|GCTInterface", target = "~", view = {class = "sifive.enterprise.grandcentral.AugmentedBundleType", defName = "ViewName", elements = [{description = "the register in GCTInterface", name = "register", tpe = {class = "sifive.enterprise.grandcentral.AugmentedBundleType", defName = "Register", elements = [{name = "_2", tpe = {class = "sifive.enterprise.grandcentral.AugmentedVectorType", elements = [{class = "sifive.enterprise.grandcentral.AugmentedGroundType", ref = {circuit = "GCTInterface", component = [{class = "firrtl.annotations.TargetToken$Field", value = "_2"}, {class = "firrtl.annotations.TargetToken$Index", value = 0 : i64}], module = "GCTInterface", path = [], ref = "r"}, tpe = {class = "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"}}, {class = "sifive.enterprise.grandcentral.AugmentedGroundType", ref = {circuit = "GCTInterface", component = [{class = "firrtl.annotations.TargetToken$Field", value = "_2"}, {class = "firrtl.annotations.TargetToken$Index", value = 1 : i64}], module = "GCTInterface", path = [], ref = "r"}, tpe = {class = "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"}}]}}, {name = "_0_inst", tpe = {class = "sifive.enterprise.grandcentral.AugmentedBundleType", defName = "_0_def", elements = [{name = "_1", tpe = {class = "sifive.enterprise.grandcentral.AugmentedGroundType", ref = {circuit = "GCTInterface", component = [{class = "firrtl.annotations.TargetToken$Field", value = "_0"}, {class = "firrtl.annotations.TargetToken$Field", value = "_1"}], module = "GCTInterface", path = [], ref = "r"}, tpe = {class = "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"}}}, {name = "_0", tpe = {class = "sifive.enterprise.grandcentral.AugmentedGroundType", ref = {circuit = "GCTInterface", component = [{class = "firrtl.annotations.TargetToken$Field", value = "_0"}, {class = "firrtl.annotations.TargetToken$Field", value = "_0"}], module = "GCTInterface", path = [], ref = "r"}, tpe = {class = "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"}}}]}}]}}, {description = "the port 'a' in GCTInterface", name = "port", tpe = {class = "sifive.enterprise.grandcentral.AugmentedGroundType", ref = {circuit = "GCTInterface", component = [], module = "GCTInterface", path = [], ref = "a"}, tpe = {class = "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"}}}]}}, {target = "~", unrelatedAnnotation}]}  {
  firrtl.module @view_companion() {
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
    // CHECK-SAME: {unrelatedAnnotation}

    // The companion should be marked.
    // CHECK: firrtl.module @view_companion
    // CHECK-SAME: annotations
    // CHECK-SAME: {class = "sifive.enterprise.grandcentral.ViewAnnotation",
    // CHECK-SAME:  id = [[ID_ViewName]] : i64,
    // CHECK-SAME:  type = "companion"}

    // The parent should be annotated. Additionally, this example has all the
    // members of the interface inside the parent.  Both port "a" and register
    // "r" should be annotated.
    // CHECK: firrtl.module @GCTInterface
    // CHECK-SAME: %a: !firrtl.uint<1> [
    // CHECK-SAME:   #firrtl.subAnno<fieldID = 0, {
    // CHECK-SAME:     class = "sifive.enterprise.grandcentral.AugmentedGroundType",
    // CHECK-SAME:     id = [[ID_port]] : i64}>
    // CHECK-SAME:   #firrtl.subAnno<fieldID = 0, {
    // CHECK-SAME:     class = "firrtl.transforms.DontTouchAnnotation"}>]
    // CHECK-SAME: annotations = [
    // CHECK-SAME:   {class = "sifive.enterprise.grandcentral.ViewAnnotation",
    // CHECK-SAME:    id = [[ID_ViewName]] : i64,
    // CHECK-SAME:    name = "view",
    // CHECK-SAME:    type = "parent"}]
    // CHECK: firrtl.reg
    // CHECK-SAME: annotations
    // CHECK-SAME: #firrtl.subAnno<fieldID = 5,
    // CHECK-SAME:   {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
    // CHECK-SAME:     id = [[ID_2_0]] : i64}>
    // CHECK-SAME: #firrtl.subAnno<fieldID = 6,
    // CHECK-SAME:   {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
    // CHECK-SAME:     id = [[ID_2_1]] : i64}>
    // CHECK-SAME: #firrtl.subAnno<fieldID = 3,
    // CHECK-SAME:    {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
    // CHECK-SAME:     id = [[ID_1]] : i64}>
    // CHECK-SAME: #firrtl.subAnno<fieldID = 2,
    // CHECK-SAME:    {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
    // CHECK-SAME:     id = [[ID_0]] : i64}>

// -----

// Test weird Grand Central AugmentedTypes which do not have a mapping in the
// Verilog.  This test is primarily making sure that these don't error.


firrtl.circuit "Foo"  attributes {raw_annotations = [{class = "sifive.enterprise.grandcentral.ViewAnnotation", companion = "~Foo|Bar_companion", name = "Bar", parent = "~Foo|Foo", target = "~", view = {class = "sifive.enterprise.grandcentral.AugmentedBundleType", defName = "View", elements = [{description = "a string", name = "string", tpe = {class = "sifive.enterprise.grandcentral.AugmentedStringType", value = "hello"}}, {description = "a boolean", name = "boolean", tpe = {class = "sifive.enterprise.grandcentral.AugmentedBooleanType", value = false}}, {description = "an integer", name = "integer", tpe = {class = "sifive.enterprise.grandcentral.AugmentedIntegerType", value = 42 : i64}}, {description = "a double", name = "double", tpe = {class = "sifive.enterprise.grandcentral.AugmentedDoubleType", value = 3.140000e+00 : f64}}]}}]}  {
  firrtl.extmodule @Bar_companion()
  firrtl.module @Foo() {
    firrtl.instance Bar_companion  @Bar_companion()
  }
}
    // CHECK-LABEL: firrtl.circuit "Foo"
    // CHECK-SAME: annotations = [{class = "[[_:.+]]AugmentedBundleType", [[_:.+]] elements = [{
    // CHECK-SAME: "sifive.enterprise.grandcentral.AugmentedStringType"
    // CHECK-SAME: "sifive.enterprise.grandcentral.AugmentedBooleanType"
    // CHECK-SAME: "sifive.enterprise.grandcentral.AugmentedIntegerType"
    // CHECK-SAME: "sifive.enterprise.grandcentral.AugmentedDoubleType"

// -----

// Multiple non-local Annotations are supported.

firrtl.circuit "Foo"  attributes {raw_annotations = [{a = "a", target = "~Foo|Foo/bar:Bar/baz:Baz"}, {b = "b", target = "~Foo|Foo/bar:Bar/baz:Baz"}]}  {
  firrtl.module @Baz() {
    firrtl.skip
  }
  firrtl.module @Bar() {
    firrtl.instance baz  @Baz()
  }
  firrtl.module @Foo() {
    firrtl.instance bar  @Bar()
  }
}

// CHECK-LABEL: firrtl.circuit "Foo"
// CHECK: firrtl.nla @nla_2 [@Foo, @Bar, @Baz] ["bar", "baz", "Baz"]
// CHECK: firrtl.nla @nla_1 [@Foo, @Bar, @Baz] ["bar", "baz", "Baz"]
// CHECK: firrtl.module @Baz
// CHECK-SAME: annotations = [{a = "a", circt.nonlocal = @nla_1}, {b = "b", circt.nonlocal = @nla_2}]
// CHECK: firrtl.module @Bar()
// CHECK: firrtl.instance baz
// CHECK-SAME: [{circt.nonlocal = @nla_1, class = "circt.nonlocal"}, {circt.nonlocal = @nla_2, class = "circt.nonlocal"}]
// CHECK: firrtl.module @Foo()
// CHECK: firrtl.instance bar
// CHECK-SAME: [{circt.nonlocal = @nla_1, class = "circt.nonlocal"}, {circt.nonlocal = @nla_2, class = "circt.nonlocal"}]

// -----

// Grand Central's SignalDriverAnnotation is properly scattered to the circuit
// and the targeted operations.

firrtl.circuit "Sub"  attributes {raw_annotations = [{annotations = [], circuit = "", circuitPackage = "other", class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", sinkTargets = [{_1 = "~Top|Foo>clock", _2 = "~Sub|Sub>clockSink"}, {_1 = "~Top|Foo>dataIn.a.b.c", _2 = "~Sub|Sub>dataSink.u"}, {_1 = "~Top|Foo>dataIn.d", _2 = "~Sub|Sub>dataSink.v"}, {_1 = "~Top|Foo>dataIn.e", _2 = "~Sub|Sub>dataSink.w"}], sourceTargets = [{_1 = "~Top|Top>clock", _2 = "~Sub|Sub>clockSource"}, {_1 = "~Top|Foo>dataOut.x.y.z", _2 = "~Sub|Sub>dataSource.u"}, {_1 = "~Top|Foo>dataOut.w", _2 = "~Sub|Sub>dataSource.v"}, {_1 = "~Top|Foo>dataOut.p", _2 = "~Sub|Sub>dataSource.w"}], target = "~"}]}  {
  firrtl.extmodule @SubExtern(in clockIn: !firrtl.clock, out clockOut: !firrtl.clock, in someInput: !firrtl.bundle<u: uint<42>, v: uint<9001>, w: vector<uint<1>, 2>>, out someOutput: !firrtl.bundle<u: uint<42>, v: uint<9001>, w: vector<uint<1>, 2>>)
  firrtl.module @Sub() {
    %clockSource = firrtl.wire  : !firrtl.clock
    %clockSink = firrtl.wire  : !firrtl.clock
    %dataSource = firrtl.wire  : !firrtl.bundle<u: uint<42>, v: uint<9001>, w: vector<uint<1>, 2>>
    %dataSink = firrtl.wire  : !firrtl.bundle<u: uint<42>, v: uint<9001>, w: vector<uint<1>, 2>>
    %ext_clockIn, %ext_clockOut, %ext_someInput, %ext_someOutput = firrtl.instance ext  @SubExtern(in clockIn: !firrtl.clock, out clockOut: !firrtl.clock, in someInput: !firrtl.bundle<u: uint<42>, v: uint<9001>, w: vector<uint<1>, 2>>, out someOutput: !firrtl.bundle<u: uint<42>, v: uint<9001>, w: vector<uint<1>, 2>>)
    firrtl.connect %ext_clockIn, %clockSource : !firrtl.clock, !firrtl.clock
    firrtl.connect %ext_someInput, %dataSource : !firrtl.bundle<u: uint<42>, v: uint<9001>, w: vector<uint<1>, 2>>, !firrtl.bundle<u: uint<42>, v: uint<9001>, w: vector<uint<1>, 2>>
    firrtl.connect %clockSink, %ext_clockOut : !firrtl.clock, !firrtl.clock
    firrtl.connect %dataSink, %ext_someOutput : !firrtl.bundle<u: uint<42>, v: uint<9001>, w: vector<uint<1>, 2>>, !firrtl.bundle<u: uint<42>, v: uint<9001>, w: vector<uint<1>, 2>>
  }
}

// CHECK-LABEL: firrtl.circuit "Sub"
// CHECK-SAME: {annotations = [], circuit = "", circuitPackage = "other", class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", id = [[ID:.+]] : i64}

// CHECK-LABEL: firrtl.module @Sub
// CHECK-SAME: {class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", id = [[ID]] : i64}

// CHECK: %clockSource = firrtl.wire
// CHECK-SAME: {class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", dir = "source", id = [[ID]] : i64, peer = "~Top|Top>clock", side = "local", targetId = 1 : i64}
// CHECK: %clockSink = firrtl.wire
// CHECK-SAME: {class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", dir = "sink", id = [[ID]] : i64, peer = "~Top|Foo>clock", side = "local", targetId = 5 : i64}

// CHECK: %dataSource = firrtl.wire
// CHECK-SAME: #firrtl<"subAnno<fieldID = 1, {class = \22sifive.enterprise.grandcentral.SignalDriverAnnotation\22, dir = \22source\22, id = [[ID]] : i64, peer = \22~Top|Foo>dataOut.x.y.z\22, side = \22local\22, targetId = 2 : i64}>">
// CHECK-SAME: #firrtl<"subAnno<fieldID = 2, {class = \22sifive.enterprise.grandcentral.SignalDriverAnnotation\22, dir = \22source\22, id = [[ID]] : i64, peer = \22~Top|Foo>dataOut.w\22, side = \22local\22, targetId = 3 : i64}>">
// CHECK-SAME: #firrtl<"subAnno<fieldID = 3, {class = \22sifive.enterprise.grandcentral.SignalDriverAnnotation\22, dir = \22source\22, id = [[ID]] : i64, peer = \22~Top|Foo>dataOut.p\22, side = \22local\22, targetId = 4 : i64}>">

// CHECK: %dataSink = firrtl.wire
// CHECK-SAME: #firrtl<"subAnno<fieldID = 1, {class = \22sifive.enterprise.grandcentral.SignalDriverAnnotation\22, dir = \22sink\22, id = [[ID]] : i64, peer = \22~Top|Foo>dataIn.a.b.c\22, side = \22local\22, targetId = 6 : i64}>">
// CHECK-SAME: #firrtl<"subAnno<fieldID = 2, {class = \22sifive.enterprise.grandcentral.SignalDriverAnnotation\22, dir = \22sink\22, id = [[ID]] : i64, peer = \22~Top|Foo>dataIn.d\22, side = \22local\22, targetId = 7 : i64}>">
// CHECK-SAME: #firrtl<"subAnno<fieldID = 3, {class = \22sifive.enterprise.grandcentral.SignalDriverAnnotation\22, dir = \22sink\22, id = [[ID]] : i64, peer = \22~Top|Foo>dataIn.e\22, side = \22local\22, targetId = 8 : i64}>">

// -----

// Grand Central's ModuleReplacementAnnotation is properly scattered to the circuit
// and the targeted operations.

firrtl.circuit "Top"  attributes {raw_annotations = [{annotations = ["foo", "bar"], circuit = "", circuitPackage = "other", class = "sifive.enterprise.grandcentral.ModuleReplacementAnnotation", dontTouches = ["~Top|Child>in", "~Top|Child>out"], target = "~", targets = ["~Top|Top/child:Child", "~Top|Top/childWrapper:ChildWrapper/child:Child"]}]}  {
  firrtl.module @ChildWrapper(in %in: !firrtl.uint<123>, out %out: !firrtl.uint<456>) {
    %child_in, %child_out = firrtl.instance child  @Child(in in: !firrtl.uint<123>, out out: !firrtl.uint<456>)
    firrtl.connect %child_in, %in : !firrtl.uint<123>, !firrtl.uint<123>
    firrtl.connect %out, %child_out : !firrtl.uint<456>, !firrtl.uint<456>
  }
  firrtl.extmodule @Child(in in: !firrtl.uint<123>, out out: !firrtl.uint<456>)
  firrtl.module @Top() {
    %child_in, %child_out = firrtl.instance child  @Child(in in: !firrtl.uint<123>, out out: !firrtl.uint<456>)
    %childWrapper_in, %childWrapper_out = firrtl.instance childWrapper  @ChildWrapper(in in: !firrtl.uint<123>, out out: !firrtl.uint<456>)
  }
}

// CHECK-LABEL: firrtl.circuit "Top"
// CHECK-SAME: {annotations = ["foo", "bar"], circuit = "", circuitPackage = "other", class = "sifive.enterprise.grandcentral.ModuleReplacementAnnotation", id = [[ID:.+]] : i64}

// CHECK: %child_in, %child_out = firrtl.instance child
// CHECK-SAME: {annotations = [{circt.nonlocal = @"~Top|Top/childWrapper:ChildWrapper/child:Child", class = "circt.nonlocal"}]}

// CHECK: firrtl.extmodule @Child(
// CHECK-SAME:   in in: !firrtl.uint<123> [{class = "firrtl.transforms.DontTouchAnnotation"}],
// CHECK-SAME:   out out: !firrtl.uint<456> [{class = "firrtl.transforms.DontTouchAnnotation"}]
// CHECK-SAME: )
// CHECK-SAME: attributes {annotations = [
// CHECK-SAME:   {circt.nonlocal = @"~Top|Top/child:Child", id = [[ID]] : i64},
// CHECK-SAME:   {circt.nonlocal = @"~Top|Top/childWrapper:ChildWrapper/child:Child", id = [[ID]] : i64}
// CHECK-SAME: ]}
