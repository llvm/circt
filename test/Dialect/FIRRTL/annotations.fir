; RUN: circt-translate -import-firrtl -split-input-file %s | FileCheck %s

; An Annotation without a target is promoted to a CircuitTarget Annotation. This
; represents a Scala FIRRTL Compiler `firrtl.annotations.NoTargetAnnotation`.
circuit Foo: %[[{"a":"a"}]]
  module Foo:
    skip

    ; CHECK-LABEL: firrtl.circuit "Foo" attributes {annotations = [{a = "a"}]}

; // -----

; A legacy `firrtl.annotations.CircuitName` annotation becomes a CircuitTarget
; Annotation.
circuit Foo: %[[{"a":"a","target":"Foo"}]]
  module Foo:
    skip

    ; CHECK-LABEL: firrtl.circuit "Foo" attributes {annotations = [{a = "a"}]}

; // -----

; A CircuitTarget Annotation is attached to the circuit.
circuit Foo: %[[{"a":"a","target":"~Foo"}]]
  module Foo:
    skip

    ; CHECK-LABEL: firrtl.circuit "Foo" attributes {annotations = [{a = "a"}]}

; // -----

; A legacy `firrtl.annotations.ModuleName` annotation becomes a ModuleTarget
; Annotation
circuit Foo: %[[{"a":"a","target":"Foo.Foo"}]]
  module Foo:
    skip

    ; CHECK-LABEL: firrtl.circuit "Foo"
    ; CHECK: firrtl.module @Foo() attributes {annotations = [{a = "a"}]}

; // -----

; A ModuleTarget Annotation is attached to the correct module.
circuit Foo: %[[{"a":"a","target":"~Foo|Foo"}]]
  module Foo:
    skip

    ; CHECK-LABEL: firrtl.circuit "Foo" {
    ; CHECK: firrtl.module @Foo() attributes {annotations = [{a = "a"}]}

; // -----

; A ModuleTarget Annotation can be attached to an ExtModule.
circuit Foo: %[[{"a":"a","target":"~Foo|Bar"}]]
  extmodule Bar:
    input a: UInt<1>
  module Foo:
    input a: UInt<1>
    inst bar of Bar
    bar.a <= a

    ; CHECK-LABEL: firrtl.circuit "Foo" {
    ; CHECK: firrtl.extmodule @Bar
    ; CHECK-SAME: attributes {annotations = [{a = "a"}]}

; // -----

; A ReferenceTarget, ComponentName, or InstanceTarget pointing at an Instance
; should work.
circuit Foo: %[[
{"a":"a","target":"~Foo|Foo>bar"},
{"b":"b","target":"Foo.Foo.bar"},
{"c":"c","target":"~Foo|Foo/bar:Bar"}
]]
  module Bar:
    skip
  module Foo:
    inst bar of Bar
    ; CHECK-LABEL: firrtl.circuit "Foo"
    ; CHECK: firrtl.nla  @nla_1 [@Foo, @Bar] ["bar", "Bar"]
    ; CHECK: firrtl.module @Bar
    ; CHECK-SAME annotations = [{c = "c"}]
    ; CHECK: firrtl.module @Foo
    ; CHECK: firrtl.instance bar
    ; CHECK-SAME: annotations = [{a = "a"}, {b = "b"}, {circt.nonlocal = @nla_1, class = "circt.nonlocal"}]

; // -----

; Test result annotations of InstanceOp.
circuit Foo: %[[{"one":null,"target":"~Foo|Foo>bar.a"},
                {"two":null,"target":"~Foo|Foo>bar.b.baz"},
                {"three":null,"target":"~Foo|Foo/bar:Bar>b.qux"},
                {"four":null,"target":"Foo.Foo.bar.c"}]]
  module Bar:
    input a: UInt<1>
    output b: {baz: UInt<1>, qux: UInt<1>}
    output c: UInt<1>
  module Foo:
    inst bar of Bar

    ; CHECK-LABEL: firrtl.circuit "Foo"
    ; CHECK: firrtl.nla @nla_1 [@Foo, @Bar] ["bar", "b"]
    ; CHECK: firrtl.module @Bar
    ; CHECK-SAME: [#firrtl.subAnno<fieldID = 2, {circt.nonlocal = @nla_1, three}>]
    ; CHECK: %bar_a, %bar_b, %bar_c = firrtl.instance bar
    ; CHECK-SAME: [{one}],
    ; CHECK-SAME: [#firrtl.subAnno<fieldID = 1, {two}>],
    ; CHECK-SAME: [{four}]

; // -----

; A ReferenceTarget/ComponentName pointing at a CombMem should work.
circuit Foo: %[[{"a":"a","target":"~Foo|Foo>bar"},{"b":"b","target":"Foo.Foo.bar"}]]
  module Foo:
    cmem bar: UInt<1>[8]

    ; CHECK-LABEL: module {
    ; CHECK: firrtl.combmem
    ; CHECK-SAME: annotations = [{a = "a"}, {b = "b"}]

; // -----

; A ReferenceTarget/ComponentName pointing at a memory should work.
circuit Foo: %[[{"a":"a","target":"~Foo|Foo>bar"},{"b":"b","target":"Foo.Foo.bar"}]]
  module Foo:
    mem bar:
      data-type => UInt<8>
      depth => 16
      reader => r
      writer => w
      read-latency => 0
      write-latency => 1
      read-under-write => undefined

    ; CHECK-LABEL: module {
    ; CHECK: firrtl.mem
    ; CHECK-SAME: annotations = [{a = "a"}, {b = "b"}]

; // -----

; Test result annotations of MemOp.
circuit Foo: %[[{"a":null,"target":"~Foo|Foo>bar.r"},
                {"b":null,"target":"~Foo|Foo>bar.r.data.baz"},
                {"c":null,"target":"~Foo|Foo>bar.w.en"},
                {"d":null,"target":"~Foo|Foo>bar.w.data.qux"}]]
  module Foo:
    mem bar:
      data-type => {baz: UInt<8>, qux: UInt<8>}
      depth => 16
      reader => r
      writer => w
      read-latency => 0
      write-latency => 1
      read-under-write => undefined

    ; CHECK-LABEL: module {
    ; CHECK: firrtl.mem
    ; CHECK-SAME: portAnnotations = [
    ; CHECK-SAME: [{a}, #firrtl.subAnno<fieldID = 5, {b}>],
    ; CHECK-SAME: [#firrtl.subAnno<fieldID = 2, {c}>, #firrtl.subAnno<fieldID = 6, {d}>]

; // -----

; A ReferenceTarget/ComponentName pointing at a node should work.  This
; shouldn't crash if the node is in a nested block.
circuit Foo: %[[{"a":"a","target":"~Foo|Foo>bar"},{"b":"b","target":"Foo.Foo.baz"}]]
  module Foo:
    input cond: UInt<1>[2]
    node bar = UInt<1>(0)
    when cond[0]:
      when cond[1]:
        node baz = UInt<1>(0)

    ; CHECK-LABEL: module {
    ; CHECK: %bar = firrtl.node
    ; CHECK-SAME: annotations = [{a = "a"}
    ; CHECK: %baz = firrtl.node
    ; CHECK-SAME: annotations = [{b = "b"}]

; // -----

; A ReferenceTarget/ComponentName pointing at a wire should work.
circuit Foo: %[[{"a":"a","target":"~Foo|Foo>bar"},{"b":"b","target":"Foo.Foo.bar"}]]
  module Foo:
    wire bar: UInt<1>

    ; CHECK-LABEL: module {
    ; CHECK: %bar = firrtl.wire
    ; CHECK-SAME: annotations = [{a = "a"}, {b = "b"}]

; // -----

; A ReferenceTarget/ComponentName pointing at a register should work.
circuit Foo: %[[{"a":"a","target":"~Foo|Foo>bar"},{"b":"b","target":"Foo.Foo.baz"}]]
  module Foo:
    input clock: Clock
    input reset: UInt<1>
    reg bar: UInt<1>, clock
    reg baz: UInt<1>, clock with : (reset => (reset, UInt<1>(0)))

    ; CHECK-LABEL: module {
    ; CHECK: %bar = firrtl.reg
    ; CHECK-SAME: annotations = [{a = "a"}]
    ; CHECK: %baz = firrtl.regreset
    ; CHECK-SAME: annotations = [{b = "b"}]

; // -----

; A ReferenceTarget/ComponentName pointing at an SeqMem should work.
circuit Foo: %[[{"a":"a","target":"~Foo|Foo>bar"},{"b":"b","target":"Foo.Foo.bar"}]]
  module Foo:
    smem bar: UInt<1>[8]

    ; CHECK-LABEL: module {
    ; CHECK: firrtl.seqmem
    ; CHECK-SAME: annotations = [{a = "a"}, {b = "b"}]

; // -----

; A ReferenceTarget/ComponentName pointing at a module/extmodule port should work.
circuit Foo: %[[{"a":"a","target":"~Foo|Bar>bar"},{"b":"b","target":"Foo.Foo.foo"}]]
  extmodule Bar:
    input bar: UInt<1>
  module Foo:
    input foo: UInt<1>
    inst bar of Bar
    bar.bar <= foo

    ; CHECK-LABEL: module {
    ; CHECK: firrtl.extmodule @Bar
    ; CHECK-SAME: [[_:.+]] [{a = "a"}]
    ; CHECK: firrtl.module @Foo
    ; CHECK-SAME: %foo: [[_:.+]] [{b = "b"}]

; // -----

; All types of JSON values should work
circuit Foo: %[[
  {
    "string": "a",
     "integer": 42,
    "float": 3.14,
    "boolean": true,
    "null": null,
    "object": {
      "foo": "bar"
    },
    "array": [1, 2, 3]
  }
]]
  module Foo:
    skip

    ; CHECK-LABEL: module {
    ; CHECK: firrtl.circuit "Foo" attributes {annotations =
    ; CHECK-SAME: array = [1, 2, 3]
    ; CHECK-SAME: boolean = true
    ; CHECK-SAME: float = 3.140
    ; CHECK-SAME: integer = 42
    ; CHECK-SAME: object = {foo = "bar"}
    ; CHECK-SAME: string = "a"

; // -----

; JSON escapes should work.
circuit Foo: %[[{"\"":"}]]"}]]
  module Foo:
    skip

    ; CHECK-LABEL: module {
    ; CHECK: firrtl.circuit "Foo" attributes {annotations =

; // -----
; JSON with a JSON-quoted string should be expanded.
circuit Foo: %[[{"a":"{\"b\":null}"}]]
  module Foo:
    skip

    ; CHECK-LABEL: module {
    ; CHECK: firrtl.circuit "Foo" attributes {annotations = [{a = {b}}]}

; // -----

; Subfield/Subindex annotations should be parsed correctly on wires
circuit Foo: %[[{"one":null,"target":"~Foo|Foo>bar[0]"},{"two":null,"target":"~Foo|Foo>bar[1].baz"} ]]
  module Foo:
    wire bar: {baz: UInt<1>, qux: UInt<1>}[2]

    ; CHECK-LABEL: module {
    ; CHECK: %bar = firrtl.wire  {annotations =
    ; CHECK-SAME: #firrtl.subAnno<fieldID = 1, {one}>
    ; CHECK-SAME: #firrtl.subAnno<fieldID = 5, {two}>


; // -----

; Subfield/Subindex annotations should be parsed correctly on registers
circuit Foo: %[[{"one":null,"target":"~Foo|Foo>bar[0]"},{"two":null,"target":"~Foo|Foo>bar[1].baz"} ]]
  module Foo:
    input clock: Clock
    reg bar: {baz: UInt<1>, qux: UInt<1>}[2], clock

    ; CHECK-LABEL: module {
    ; CHECK: %bar = firrtl.reg %clock  {annotations =
    ; CHECK-SAME: #firrtl.subAnno<fieldID = 1, {one}>
    ; CHECK-SAME: #firrtl.subAnno<fieldID = 5, {two}>

; // -----

; Subindices should not get sign-extended and cause problems.  This circuit has
; caused bugs in the past.
circuit Foo: %[[{"a":null,"target":"~Foo|Foo>w[9]"}]]
  module Foo:
    input a: UInt<1>[18]
    output b: UInt<1>[18]

    wire w: UInt<1>[18]

    w <= a
    b <= w

    ; CHECK-LABEL: module {
    ; CHECK: %w = firrtl.wire {annotations =
    ; CHECK-SAME: #firrtl.subAnno<fieldID = 10, {a}

; // -----

; Annotations should apply even when the target's name is dropped.
circuit Foo: %[[{"target": "~Foo|Foo>_T_0", "a": "a"},
                {"target": "~Foo|Foo>_T_1", "a": "a"},
                {"target": "~Foo|Foo>_T_2", "a": "a"},
                {"target": "~Foo|Foo>_T_3", "a": "a"},
                {"target": "~Foo|Foo>_T_4", "a": "a"},
                {"target": "~Foo|Foo>_T_5", "a": "a"},
                {"target": "~Foo|Foo>_T_6", "a": "a"},
                {"target": "~Foo|Foo>_T_7", "a": "a"},
                {"target": "~Foo|Foo>_T_8", "a": "a"}]]
  module Bar:
    skip
  module Foo:
    input reset : UInt<1>
    input clock : Clock

    ; CHECK: %_T_0 = firrtl.wire  {annotations = [{a = "a"}]}
    wire _T_0 : UInt<1>
    ; CHECK: %_T_1 = firrtl.node
    node _T_1 = _T_0
    ; CHECK: %_T_2 = firrtl.reg %clock  {annotations = [{a = "a"}]}
    reg _T_2 : UInt<1>, clock
    ; CHECK: %_T_3 = firrtl.regreset {{.+}}  {annotations = [{a = "a"}]}
    reg _T_3 : UInt<4>, clock with :
      reset => (reset, UInt<4>("h0"))
    ; CHECK: %_T_4 = firrtl.seqmem Undefined {annotations = [{a = "a"}]}
    smem _T_4 : UInt<1>[9] [256]
    ; CHECK: %_T_5 = firrtl.combmem  {annotations = [{a = "a"}]}
    cmem _T_5 : UInt<1>[9] [256]
    ; CHECK: firrtl.memoryport {{.+}} {annotations = [{a = "a"}]
    infer mport _T_6 = _T_5[reset], clock
    ; CHECK: firrtl.instance _T_7 {annotations = [{a = "a"}]}
    inst _T_7 of Bar
    ; CHECK: firrtl.mem Undefined  {annotations = [{a = "a"}]
    mem _T_8 :
        data-type => UInt<4>
        depth => 8
        writer => w
        read-latency => 0
        write-latency => 1
        read-under-write => undefined

; // -----

; DontTouch annotation preserves temporary names
circuit Foo: %[[{"target": "~Foo|Foo>_T_0", "class": "firrtl.transforms.DontTouchAnnotation"},
                {"target": "~Foo|Foo>_T_1", "class": "firrtl.transforms.DontTouchAnnotation"},
                {"target": "~Foo|Foo>_T_2", "class": "firrtl.transforms.DontTouchAnnotation"},
                {"target": "~Foo|Foo>_T_3", "class": "firrtl.transforms.DontTouchAnnotation"},
                {"target": "~Foo|Foo>_T_4", "class": "firrtl.transforms.DontTouchAnnotation"},
                {"target": "~Foo|Foo>_T_5", "class": "firrtl.transforms.DontTouchAnnotation"},
                {"target": "~Foo|Foo>_T_6", "class": "firrtl.transforms.DontTouchAnnotation"},
                {"target": "~Foo|Foo>_T_7", "class": "firrtl.transforms.DontTouchAnnotation"},
                {"target": "~Foo|Foo>_T_8", "class": "firrtl.transforms.DontTouchAnnotation"},
                {"target": "~Foo|Foo>_T_9.a", "class": "firrtl.transforms.DontTouchAnnotation"}]]
  module Bar:
    skip
  module Foo:
    input reset : UInt<1>
    input clock : Clock

    ; CHECK: %_T_0 = firrtl.wire  {annotations =
    ; CHECK-SAME: {class = "firrtl.transforms.DontTouchAnnotation"}
    wire _T_0 : UInt<1>
    ; CHECK: %_T_1 = firrtl.node %_T_0  {annotations =
    ; CHECK-SAME: {class = "firrtl.transforms.DontTouchAnnotation"}
    node _T_1 = _T_0
    ; CHECK: %_T_2 = firrtl.reg %clock  {annotations =
    ; CHECK-SAME: {class = "firrtl.transforms.DontTouchAnnotation"}
    reg _T_2 : UInt<1>, clock
    ; CHECK: %_T_3 = firrtl.regreset %clock, %reset, %c0_ui4  {annotations =
    ; CHECK-SAME: {class = "firrtl.transforms.DontTouchAnnotation"}
    reg _T_3 : UInt<4>, clock with :
      reset => (reset, UInt<4>("h0"))
    ; CHECK: %_T_4 = firrtl.seqmem Undefined  {annotations =
    ; CHECK-SAME: {class = "firrtl.transforms.DontTouchAnnotation"}
    smem _T_4 : UInt<1>[9] [256]
    ; CHECK: %_T_5 = firrtl.combmem  {annotations = [
    ; CHECK-SAME: {class = "firrtl.transforms.DontTouchAnnotation"}
    cmem _T_5 : UInt<1>[9] [256]
    ; CHECK: firrtl.memoryport Infer %_T_5 {annotations =
    ; CHECK-SAME: {class = "firrtl.transforms.DontTouchAnnotation"}
    infer mport _T_6 = _T_5[reset], clock
    ; CHECK: firrtl.instance _T_7 {annotations =
    ; CHECK-SAME: {class = "firrtl.transforms.DontTouchAnnotation"}
    inst _T_7 of Bar
    ; CHECK: firrtl.mem Undefined  {annotations =
    ; CHECK_SAME: {class = "firrtl.transforms.DontTouchAnnotation"}
    mem _T_8 :
        data-type => UInt<4>
        depth => 8
        writer => w
        read-latency => 0
        write-latency => 1
        read-under-write => undefined

    wire aggregate : { a : UInt<1>}
    ; CHECK: %_T_9 = firrtl.node
    ; CHECK-SAME: {class = "firrtl.transforms.DontTouchAnnotation"}
    node _T_9 = aggregate


; // -----

; Test that an annotated, anonymous node is preserved if annotated.  Normally,
; the FIRRTL parser will aggressively eliminate these.
circuit AnnotationsBlockNodePruning: %[[
  {"a": null, "target": "~AnnotationsBlockNodePruning|AnnotationsBlockNodePruning>_T"}
]]
  module AnnotationsBlockNodePruning:
    input a: UInt<1>
    node _T = not(a)

    ; CHECK-LABEL: firrtl.module @AnnotationsBlockNodePruning
    ; CHECK: firrtl.node

; // -----

; --------------------------------------------------------------------------------
; SiFive-custom annotations related to the GrandCentral utility.  These
; annotations do not conform to standard SingleTarget or NoTarget format and
; need to be manually split up.
; --------------------------------------------------------------------------------

; Test sifive.enterprise.grandcentral.DataTapsAnnotation with all possible
; variants of DataTapKeys.
circuit GCTDataTap : %[
[
  {
    "class": "sifive.enterprise.grandcentral.DataTapsAnnotation",
    "blackBox": "~GCTDataTap|DataTap",
    "keys": [
      {
        "class": "sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source": "~GCTDataTap|GCTDataTap>r",
        "portName": "~GCTDataTap|DataTap>_0"
      },
      {
        "class": "sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source": "~GCTDataTap|GCTDataTap>r",
        "portName": "~GCTDataTap|DataTap>_1[0]"
      },
      {
        "class": "sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source": "~GCTDataTap|GCTDataTap>w.a",
        "portName": "~GCTDataTap|DataTap>_2"
      },
      {
        "class": "sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source": "~GCTDataTap|GCTDataTap>w.a",
        "portName": "~GCTDataTap|DataTap>_3[0]"
      },
      {
        "class":"sifive.enterprise.grandcentral.DataTapModuleSignalKey",
        "module":"~GCTDataTap|BlackBox",
        "internalPath":"baz.qux",
        "portName":"~GCTDataTap|DataTap>_4"
      },
      {
        "class":"sifive.enterprise.grandcentral.DataTapModuleSignalKey",
        "module":"~GCTDataTap|BlackBox",
        "internalPath":"baz.quz",
        "portName":"~GCTDataTap|DataTap>_5[0]"
      },
      {
        "class":"sifive.enterprise.grandcentral.DeletedDataTapKey",
        "portName":"~GCTDataTap|DataTap>_6"
      },
      {
        "class":"sifive.enterprise.grandcentral.DeletedDataTapKey",
        "portName":"~GCTDataTap|DataTap>_7[0]"
      },
      {
        "class":"sifive.enterprise.grandcentral.LiteralDataTapKey",
        "literal":"UInt<16>(\"h2a\")",
        "portName":"~GCTDataTap|DataTap>_8"
      },
      {
        "class":"sifive.enterprise.grandcentral.LiteralDataTapKey",
        "literal":"UInt<16>(\"h2a\")",
        "portName":"~GCTDataTap|DataTap>_9[0]"
      },
      {
        "class": "sifive.enterprise.grandcentral.ReferenceDataTapKey",
        "source": "~GCTDataTap|GCTDataTap/im:InnerMod>w",
        "portName": "~GCTDataTap|DataTap>_10"
      }
    ]
  },
  {
    "unrelatedAnnotation": null
  }
]]
  extmodule DataTap :
    output _0 : UInt<1>
    output _1 : UInt<1>[1]
    output _2 : UInt<1>
    output _3 : UInt<1>[1]
    output _4 : UInt<1>
    output _5 : UInt<1>[1]
    output _6 : UInt<1>
    output _7 : UInt<1>[1]
    output _8 : UInt<1>
    output _9 : UInt<1>[1]
    output _10 : UInt<1>

    defname = DataTap

  extmodule BlackBox:
    defname = BlackBox

  module InnerMod :

    wire w : UInt<1>

  module GCTDataTap :
    input clock : Clock
    input reset : UInt<1>
    input a : UInt<1>
    output b : UInt<1>

    reg r : UInt<1>, clock
    wire w : {a: UInt<1>}
    inst DataTap of DataTap
    inst BlackBox of BlackBox
    inst im of InnerMod

    ; CHECK-LABEL: firrtl.circuit "GCTDataTap"
    ; CHECK-SAME: annotations = [{unrelatedAnnotation}]
    ; CHECK:      firrtl.nla @nla_1 [@GCTDataTap, @InnerMod] ["im", "w"]
    ; CHECK: firrtl.extmodule @DataTap
    ; CHECK-SAME: _0: !firrtl.uint<1> [
    ; CHECK-SAME:   {class = "firrtl.transforms.DontTouchAnnotation"}
    ; CHECK-SAME:   {class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
    ; CHECK-SAME:    id = [[ID:[0-9]+]] : i64,
    ; CHECK-SAME:    portID = [[PORT_ID_0:[0-9]+]] : i64,
    ; CHECK-SAME:    type = "portName"}
    ; CHECK-SAME: _1: !firrtl.vector<uint<1>, 1> [
    ; CHECK-SAME:   #firrtl.subAnno<fieldID = 1,
    ; CHECK-SAME:     {class = "firrtl.transforms.DontTouchAnnotation"}>
    ; CHECK-SAME:   #firrtl.subAnno<fieldID = 1,
    ; CHECK-SAME:     {class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
    ; CHECK-SAME:      id = [[ID]] : i64,
    ; CHECK-SAME:      portID = [[PORT_ID_1:[0-9]+]] : i64,
    ; CHECK-SAME:      type = "portName"}>
    ; CHECK-SAME: _2: !firrtl.uint<1> [
    ; CHECK-SAME:   {class = "firrtl.transforms.DontTouchAnnotation"}
    ; CHECK-SAME:   {class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
    ; CHECK-SAME:    id = [[ID]] : i64,
    ; CHECK-SAME:    portID = [[PORT_ID_2:[0-9]+]] : i64,
    ; CHECK-SAME:    type = "portName"}
    ; CHECK-SAME: _3: !firrtl.vector<uint<1>, 1> [
    ; CHECK-SAME:   #firrtl.subAnno<fieldID = 1,
    ; CHECK-SAME:     {class = "firrtl.transforms.DontTouchAnnotation"}>
    ; CHECK-SAME:   #firrtl.subAnno<fieldID = 1,
    ; CHECK-SAME:     {class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
    ; CHECK-SAME:      id = [[ID]] : i64,
    ; CHECK-SAME:      portID = [[PORT_ID_3:[0-9]+]] : i64,
    ; CHECK-SAME:      type = "portName"}>
    ; CHECK-SAME: _4: !firrtl.uint<1> [
    ; CHECK-SAME:   {class = "firrtl.transforms.DontTouchAnnotation"}
    ; CHECK-SAME:   {class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey",
    ; CHECK-SAME:    id = [[ID]] : i64,
    ; CHECK-SAME:    portID = [[PORT_ID_4:[0-9]+]] : i64}
    ; CHECK-SAME: _5: !firrtl.vector<uint<1>, 1> [
    ; CHECK-SAME:   #firrtl.subAnno<fieldID = 1,
    ; CHECK-SAME:     {class = "firrtl.transforms.DontTouchAnnotation"}>
    ; CHECK-SAME:   #firrtl.subAnno<fieldID = 1,
    ; CHECK-SAME:     {class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey",
    ; CHECK-SAME:      id = [[ID]] : i64,
    ; CHECK-SAME:      portID = [[PORT_ID_5:[0-9]+]] : i64}>
    ; CHECK-SAME: _6: !firrtl.uint<1> [
    ; CHECK-SAME:   {class = "firrtl.transforms.DontTouchAnnotation"}
    ; CHECK-SAME:   {class = "sifive.enterprise.grandcentral.DeletedDataTapKey",
    ; CHECK-SAME:    id = [[ID]] : i64}
    ; CHECK-SAME: _7: !firrtl.vector<uint<1>, 1> [
    ; CHECK-SAME:   #firrtl.subAnno<fieldID = 1,
    ; CHECK-SAME:     {class = "firrtl.transforms.DontTouchAnnotation"}>
    ; CHECK-SAME:   #firrtl.subAnno<fieldID = 1,
    ; CHECK-SAME:     {class = "sifive.enterprise.grandcentral.DeletedDataTapKey",
    ; CHECK-SAME:      id = [[ID]] : i64}>
    ; CHECK-SAME: _8: !firrtl.uint<1> [
    ; CHECK-SAME:   {class = "firrtl.transforms.DontTouchAnnotation"}
    ; CHECK-SAME:   {class = "sifive.enterprise.grandcentral.LiteralDataTapKey",
    ; CHECK-SAME:    literal = "UInt<16>(\22h2a\22)"}
    ; CHECK-SAME: _9: !firrtl.vector<uint<1>, 1> [
    ; CHECK-SAME:   #firrtl.subAnno<fieldID = 1,
    ; CHECK-SAME:     {class = "firrtl.transforms.DontTouchAnnotation"}>
    ; CHECK-SAME:     {class = "sifive.enterprise.grandcentral.LiteralDataTapKey",
    ; CHECK-SAME:      literal = "UInt<16>(\22h2a\22)"}
    ; CHECK-SAME: _10: !firrtl.uint<1> [
    ; CHECK-SAME      {class = "firrtl.transforms.DontTouchAnnotation"}
    ; CHECK-SAME:     {class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = [[ID]] : i64, portID = [[PORT_ID_6:[0-9]+]] : i64, type = "portName"}
    ; CHECK-SAME: annotations = [
    ; CHECK-SAME:   {class = "sifive.enterprise.grandcentral.DataTapsAnnotation"},
    ; CHECK-SAME:   {class = "firrtl.transforms.DontTouchAnnotation"}]

    ; CHECK: firrtl.extmodule @BlackBox
    ; CHECK-SAME: annotations = [
    ; CHECK-SAME:   {class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey",
    ; CHECK-SAME:    id = [[ID]] : i64,
    ; CHECK-SAME:    internalPath = "baz.qux",
    ; CHECK-SAME:    portID = [[PORT_ID_4]] : i64}
    ; CHECK-SAME:   {class = "firrtl.transforms.DontTouchAnnotation"}
    ; CHECK-SAME:   {class = "sifive.enterprise.grandcentral.DataTapModuleSignalKey",
    ; CHECK-SAME:    id = [[ID]] : i64,
    ; CHECK-SAME:    internalPath = "baz.quz",
    ; CHECK-SAME:    portID = [[PORT_ID_5]] : i64}

    ; CHECK: firrtl.module @InnerMod
    ; CHECK-NEXT: %w = firrtl.wire
    ; CHECK-SAME: {circt.nonlocal = @nla_1, class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = 0 : i64, portID = 7 : i64, type = "source"}
    ; CHECK-SAME: {class = "firrtl.transforms.DontTouchAnnotation"}

    ; CHECK: firrtl.module @GCTDataTap
    ; CHECK: firrtl.reg
    ; CHECk-SAME: annotations =
    ; CHECK-SAME:   {class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
    ; CHECK-SAME:    id = [[ID]] : i64,
    ; CHECK-SAME:    portID = [[PORT_ID_0]] : i64,
    ; CHECK-SAME:    type = "source"}
    ; CHECK-SAME:   {class = "firrtl.transforms.DontTouchAnnotation"}

    ; CHECK: firrtl.wire
    ; CHECK-SAME: annotations =
    ; CHECK-SAME:   #firrtl.subAnno<fieldID = 1,
    ; CHECK-SAME:     {class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
    ; CHECK-SAME:      id = [[ID]] : i64,
    ; CHECK-SAME:      portID = [[PORT_ID_2]] : i64,
    ; CHECK-SAME:      type = "source"}>
    ; CHECK-SAME:   #firrtl.subAnno<fieldID = 1,
    ; CHECK-SAME:     {class = "firrtl.transforms.DontTouchAnnotation"}>
    ; CHECK-SAME:   #firrtl.subAnno<fieldID = 1,
    ; CHECK-SAME:     {class = "sifive.enterprise.grandcentral.ReferenceDataTapKey",
    ; CHECK-SAME:      id = [[ID]] : i64,
    ; CHECK-SAME:      portID = [[PORT_ID_3]] : i64,
    ; CHECK-SAME:      type = "source"}>

; // -----

; Test sifive.enterprise.grandcentral.MemTapAnnotation
circuit GCTMemTap : %[
[
  {
    "class":"sifive.enterprise.grandcentral.MemTapAnnotation",
    "taps":[
      "GCTMemTap.MemTap.mem[0]",
      "GCTMemTap.MemTap.mem[1]"
    ],
    "source":"~GCTMemTap|GCTMemTap>mem"
  },
  {
    "unrelatedAnnotation": null
  }
]]
  extmodule MemTap :
    output mem : UInt<1>[2]

    defname = MemTap

  module GCTMemTap :
    input clock : Clock
    input reset : UInt<1>

    cmem mem : UInt<1>[2]
    inst MemTap of MemTap
    MemTap.mem is invalid
    wire memTap : UInt<1>[2]
    memTap[0] <= MemTap.mem[0]
    memTap[1] <= MemTap.mem[1]

    ; CHECK-LABEL: firrtl.circuit "GCTMemTap"
    ; CHECK-SAME: annotations = [{unrelatedAnnotation}]
    ; CHECK: firrtl.extmodule @MemTap
    ; CHECK-SAME: mem: [[A:.+]] [
    ; CHECK-SAME:   #firrtl.subAnno<fieldID = 1,
    ; CHECK-SAME:     {class = "sifive.enterprise.grandcentral.MemTapAnnotation",
    ; CHECK-SAME:      id = [[ID:[0-9]+]] : i64, word = 0 : i64}>
    ; CHECK-SAME:   #firrtl.subAnno<fieldID = 2,
    ; CHECK-SAME:     {class = "sifive.enterprise.grandcentral.MemTapAnnotation",
    ; CHECK-SAME:      id = [[ID]] : i64, word = 1 : i64}>
    ; CHECK: firrtl.module @GCTMemTap
    ; CHECK: %mem = firrtl.combmem
    ; CHECK-SAME: annotations = [
    ; CHECK-SAME:   {class = "sifive.enterprise.grandcentral.MemTapAnnotation",
    ; CHECK-SAME:    id = [[ID]] : i64}]

; // -----

; Test sifive.enterprise.grandcentral.ViewAnnotation
circuit GCTInterface : %[
[
  {
    "class": "sifive.enterprise.grandcentral.GrandCentralView$SerializedViewAnnotation",
    "name": "view",
    "companion": "~GCTInterface|view_companion",
    "parent": "~GCTInterface|GCTInterface",
    "view": {
      "class": "sifive.enterprise.grandcentral.AugmentedBundleType",
      "defName": "ViewName",
      "elements": [
        {
          "name": "register",
          "description": "the register in GCTInterface",
          "tpe": {
            "class": "sifive.enterprise.grandcentral.AugmentedBundleType",
            "defName": "Register",
            "elements": [
              {
                "name": "_2",
                "tpe": {
                  "class": "sifive.enterprise.grandcentral.AugmentedVectorType",
                  "elements": [
                    {
                      "class": "sifive.enterprise.grandcentral.AugmentedGroundType",
                      "ref": {
                        "circuit": "GCTInterface",
                        "module": "GCTInterface",
                        "path": [],
                        "ref": "r",
                        "component": [
                          {
                            "class": "firrtl.annotations.TargetToken$Field",
                            "value": "_2"
                          },
                          {
                            "class": "firrtl.annotations.TargetToken$Index",
                            "value": 0
                          }
                        ]
                      },
                      "tpe": {
                        "class": "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"
                      }
                    },
                    {
                      "class": "sifive.enterprise.grandcentral.AugmentedGroundType",
                      "ref": {
                        "circuit": "GCTInterface",
                        "module": "GCTInterface",
                        "path": [],
                        "ref": "r",
                        "component": [
                          {
                            "class": "firrtl.annotations.TargetToken$Field",
                            "value": "_2"
                          },
                          {
                            "class": "firrtl.annotations.TargetToken$Index",
                            "value": 1
                          }
                        ]
                      },
                      "tpe": {
                        "class": "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"
                      }
                    }
                  ]
                }
              },
              {
                "name": "_0_inst",
                "tpe": {
                  "class": "sifive.enterprise.grandcentral.AugmentedBundleType",
                  "defName": "_0_def",
                  "elements": [
                    {
                      "name": "_1",
                      "tpe": {
                        "class": "sifive.enterprise.grandcentral.AugmentedGroundType",
                        "ref": {
                          "circuit": "GCTInterface",
                          "module": "GCTInterface",
                          "path": [],
                          "ref": "r",
                          "component": [
                            {
                              "class": "firrtl.annotations.TargetToken$Field",
                              "value": "_0"
                            },
                            {
                              "class": "firrtl.annotations.TargetToken$Field",
                              "value": "_1"
                            }
                          ]
                        },
                        "tpe": {
                          "class": "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"
                        }
                      }
                    },
                    {
                      "name": "_0",
                      "tpe": {
                        "class": "sifive.enterprise.grandcentral.AugmentedGroundType",
                        "ref": {
                          "circuit": "GCTInterface",
                          "module": "GCTInterface",
                          "path": [],
                          "ref": "r",
                          "component": [
                            {
                              "class": "firrtl.annotations.TargetToken$Field",
                              "value": "_0"
                            },
                            {
                              "class": "firrtl.annotations.TargetToken$Field",
                              "value": "_0"
                            }
                          ]
                        },
                        "tpe": {
                          "class": "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"
                        }
                      }
                    }
                  ]
                }
              }
            ]
          }
        },
        {
          "name": "port",
          "description": "the port 'a' in GCTInterface",
          "tpe": {
            "class": "sifive.enterprise.grandcentral.AugmentedGroundType",
            "ref": {
              "circuit": "GCTInterface",
              "module": "GCTInterface",
              "path": [],
              "ref": "a",
              "component": []
            },
            "tpe": {
              "class": "sifive.enterprise.grandcentral.GrandCentralView$UnknownGroundType$"
            }
          }
        }
      ]
    }
  },
  {
    "unrelatedAnnotation": null
  }
]
]
  module view_companion:
    skip
  module GCTInterface :
    input clock : Clock
    input reset : UInt<1>
    input a : UInt<1>

    reg r : {_0 : {_0 : UInt<1>, _1 : UInt<1>}, _2 : UInt<1>[2]}, clock

    inst view_companion of view_companion

    ; CHECK-LABEL: firrtl.circuit "GCTInterface"

    ; The interface definition should show up as a circuit annotation.  Nested
    ; interfaces show up as nested bundle types and not as separate interfaces.
    ; CHECK-SAME: annotations
    ; CHECK-SAME: {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
    ; CHECK-SAME:  defName = "ViewName",
    ; CHECK-SAME:  elements = [
    ; CHECK-SAME:    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
    ; CHECK-SAME:     defName = "Register",
    ; CHECK-SAME:     description = "the register in GCTInterface",
    ; CHECK-SAME:     elements = [
    ; CHECK-SAME:       {class = "sifive.enterprise.grandcentral.AugmentedVectorType",
    ; CHECK-SAME:        elements = [
    ; CHECK-SAME:          {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
    ; CHECK-SAME:           id = [[ID_2_0:[0-9]+]] : i64,
    ; CHECK-SAME:           name = "_2"},
    ; CHECK-SAME:          {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
    ; CHECK-SAME:           id = [[ID_2_1:[0-9]+]] : i64,
    ; CHECK-SAME:           name = "_2"}],
    ; CHECK-SAME:        name = "_2"},
    ; CHECK-SAME:       {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
    ; CHECK-SAME:        defName = "_0_def",
    ; CHECK-SAME:        elements = [
    ; CHECK-SAME:          {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
    ; CHECK-SAME:           id = [[ID_1:[0-9]+]] : i64,
    ; CHECK-SAME:           name = "_1"},
    ; CHECK-SAME:          {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
    ; CHECK-SAME:           id = [[ID_0:[0-9]+]] : i64,
    ; CHECK-SAME:           name = "_0"}],
    ; CHECK-SAME:        name = "_0_inst"}],
    ; CHECK-SAME:     name = "register"},
    ; CHECK-SAME:    {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
    ; CHECK-SAME:     description = "the port 'a' in GCTInterface",
    ; CHECK-SAME:     id = [[ID_port:[0-9]+]] : i64,
    ; CHECK-SAME:     name = "port"}],
    ; CHECK-SAME:  id = [[ID_ViewName:[0-9]+]] : i64,
    ; CHECK-SAME:  name = "view"}
    ; CHECK-SAME: {unrelatedAnnotation}

    ; The companion should be marked.
    ; CHECK: firrtl.module @view_companion
    ; CHECK-SAME: annotations
    ; CHECK-SAME: {class = "sifive.enterprise.grandcentral.ViewAnnotation",
    ; CHECK-SAME:  id = [[ID_ViewName]] : i64,
    ; CHECK-SAME:  type = "companion"}

    ; The parent should be annotated. Additionally, this example has all the
    ; members of the interface inside the parent.  Both port "a" and register
    ; "r" should be annotated.
    ; CHECK: firrtl.module @GCTInterface
    ; CHECK-SAME: %a: !firrtl.uint<1> [
    ; CHECK-SAME:   #firrtl.subAnno<fieldID = 0, {
    ; CHECK-SAME:     class = "sifive.enterprise.grandcentral.AugmentedGroundType",
    ; CHECK-SAME:     id = [[ID_port]] : i64}>
    ; CHECK-SAME:   #firrtl.subAnno<fieldID = 0, {
    ; CHECK-SAME:     class = "firrtl.transforms.DontTouchAnnotation"}>]
    ; CHECK-SAME: annotations = [
    ; CHECK-SAME:   {class = "sifive.enterprise.grandcentral.ViewAnnotation",
    ; CHECK-SAME:    id = [[ID_ViewName]] : i64,
    ; CHECK-SAME:    name = "view",
    ; CHECK-SAME:    type = "parent"}]
    ; CHECK: firrtl.reg
    ; CHECK-SAME: annotations
    ; CHECK-SAME: #firrtl.subAnno<fieldID = 5,
    ; CHECK-SAME:   {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
    ; CHECK-SAME:     id = [[ID_2_0]] : i64}>
    ; CHECK-SAME: #firrtl.subAnno<fieldID = 6,
    ; CHECK-SAME:   {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
    ; CHECK-SAME:     id = [[ID_2_1]] : i64}>
    ; CHECK-SAME: #firrtl.subAnno<fieldID = 3,
    ; CHECK-SAME:    {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
    ; CHECK-SAME:     id = [[ID_1]] : i64}>
    ; CHECK-SAME: #firrtl.subAnno<fieldID = 2,
    ; CHECK-SAME:    {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
    ; CHECK-SAME:     id = [[ID_0]] : i64}>


; // -----

; Test weird Grand Central AugmentedTypes which do not have a mapping in the
; Verilog.  This test is primarily making sure that these don't error.

circuit Foo : %[[
{
  "class": "sifive.enterprise.grandcentral.ViewAnnotation",
  "name": "Bar",
  "companion": "~Foo|Bar_companion",
  "parent": "~Foo|Foo",
  "view": {
    "class": "sifive.enterprise.grandcentral.AugmentedBundleType",
    "defName": "View",
    "elements": [
      {
        "name": "string",
        "description": "a string",
        "tpe": {
          "class": "sifive.enterprise.grandcentral.AugmentedStringType",
          "value": "hello"
        }
      },
      {
        "name": "boolean",
        "description": "a boolean",
        "tpe": {
          "class": "sifive.enterprise.grandcentral.AugmentedBooleanType",
          "value": false
        }
      },
      {
        "name": "integer",
        "description": "an integer",
        "tpe": {
          "class": "sifive.enterprise.grandcentral.AugmentedIntegerType",
          "value": 42
        }
      },
      {
        "name": "double",
        "description": "a double",
        "tpe": {
          "class": "sifive.enterprise.grandcentral.AugmentedDoubleType",
          "value": 3.14
        }
      }
    ]
  }
}
]]
  extmodule Bar_companion :

  module Foo :
    inst Bar_companion of Bar_companion

    ; CHECK-LABEL: firrtl.circuit "Foo"
    ; CHECK-SAME: annotations = [{class = "[[_:.+]]AugmentedBundleType", [[_:.+]] elements = [{
    ; CHECK-SAME: "sifive.enterprise.grandcentral.AugmentedStringType"
    ; CHECK-SAME: "sifive.enterprise.grandcentral.AugmentedBooleanType"
    ; CHECK-SAME: "sifive.enterprise.grandcentral.AugmentedIntegerType"
    ; CHECK-SAME: "sifive.enterprise.grandcentral.AugmentedDoubleType"

; // -----

; Multiple non-local Annotations are supported.

circuit Foo: %[[{"a":"a","target":"~Foo|Foo/bar:Bar/baz:Baz"}, {"b":"b","target":"~Foo|Foo/bar:Bar/baz:Baz"}]]
  module Baz :
    skip
  module Bar :
    inst baz of Baz
  module Foo :
    inst bar of Bar
; CHECK-LABEL: firrtl.circuit "Foo"
; CHECK: firrtl.nla @nla_2 [@Foo, @Bar, @Baz] ["bar", "baz", "Baz"]
; CHECK: firrtl.nla @nla_1 [@Foo, @Bar, @Baz] ["bar", "baz", "Baz"]
; CHECK: firrtl.module @Baz
; CHECK-SAME: annotations = [{a = "a", circt.nonlocal = @nla_1}, {b = "b", circt.nonlocal = @nla_2}]
; CHECK: firrtl.module @Bar()
; CHECK: firrtl.instance baz
; CHECK-SAME: [{circt.nonlocal = @nla_1, class = "circt.nonlocal"}, {circt.nonlocal = @nla_2, class = "circt.nonlocal"}]
; CHECK: firrtl.module @Foo()
; CHECK: firrtl.instance bar
; CHECK-SAME: [{circt.nonlocal = @nla_1, class = "circt.nonlocal"}, {circt.nonlocal = @nla_2, class = "circt.nonlocal"}]

; // -----

; Grand Central's SignalDriverAnnotation is properly scattered to the circuit
; and the targeted operations.

circuit Sub : %[[{
  "class": "sifive.enterprise.grandcentral.SignalDriverAnnotation",
  "sinkTargets": [
    {"_1": "~Top|Foo>clock",        "_2": "~Sub|Sub>clockSink" },
    {"_1": "~Top|Foo>dataIn.a.b.c", "_2": "~Sub|Sub>dataSink.u"},
    {"_1": "~Top|Foo>dataIn.d",     "_2": "~Sub|Sub>dataSink.v"},
    {"_1": "~Top|Foo>dataIn.e",     "_2": "~Sub|Sub>dataSink.w"}
  ],
  "sourceTargets": [
    {"_1": "~Top|Top>clock",         "_2": "~Sub|Sub>clockSource" },
    {"_1": "~Top|Foo>dataOut.x.y.z", "_2": "~Sub|Sub>dataSource.u"},
    {"_1": "~Top|Foo>dataOut.w",     "_2": "~Sub|Sub>dataSource.v"},
    {"_1": "~Top|Foo>dataOut.p",     "_2": "~Sub|Sub>dataSource.w"}
  ],
  "circuit": "",
  "annotations": [],
  "circuitPackage": "other"
}]]
  extmodule SubExtern :
    input clockIn : Clock
    output clockOut : Clock
    input someInput : { u: UInt<42>, v: UInt<9001>, w: UInt<1>[2] }
    output someOutput : { u: UInt<42>, v: UInt<9001>, w: UInt<1>[2] }

  module Sub :
    wire clockSource : Clock
    wire clockSink : Clock
    wire dataSource : { u: UInt<42>, v: UInt<9001>, w: UInt<1>[2] }
    wire dataSink : { u: UInt<42>, v: UInt<9001>, w: UInt<1>[2] }

    inst ext of SubExtern
    ext.clockIn <= clockSource
    ext.someInput <= dataSource
    clockSink <= ext.clockOut
    dataSink <= ext.someOutput

; CHECK-LABEL: firrtl.circuit "Sub"
; CHECK-SAME: {annotations = [], circuit = "", circuitPackage = "other", class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", id = [[ID:.+]] : i64}

; CHECK-LABEL: firrtl.module @Sub
; CHECK-SAME: {class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", id = [[ID]] : i64}

; CHECK: %clockSource = firrtl.wire
; CHECK-SAME: {class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", dir = "source", id = [[ID]] : i64, peer = "~Top|Top>clock", side = "local", targetId = 1 : i64}
; CHECK: %clockSink = firrtl.wire
; CHECK-SAME: {class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", dir = "sink", id = [[ID]] : i64, peer = "~Top|Foo>clock", side = "local", targetId = 5 : i64}

; CHECK: %dataSource = firrtl.wire
; CHECK-SAME: #firrtl<"subAnno<fieldID = 1, {class = \22sifive.enterprise.grandcentral.SignalDriverAnnotation\22, dir = \22source\22, id = [[ID]] : i64, peer = \22~Top|Foo>dataOut.x.y.z\22, side = \22local\22, targetId = 2 : i64}>">
; CHECK-SAME: #firrtl<"subAnno<fieldID = 2, {class = \22sifive.enterprise.grandcentral.SignalDriverAnnotation\22, dir = \22source\22, id = [[ID]] : i64, peer = \22~Top|Foo>dataOut.w\22, side = \22local\22, targetId = 3 : i64}>">
; CHECK-SAME: #firrtl<"subAnno<fieldID = 3, {class = \22sifive.enterprise.grandcentral.SignalDriverAnnotation\22, dir = \22source\22, id = [[ID]] : i64, peer = \22~Top|Foo>dataOut.p\22, side = \22local\22, targetId = 4 : i64}>">

; CHECK: %dataSink = firrtl.wire
; CHECK-SAME: #firrtl<"subAnno<fieldID = 1, {class = \22sifive.enterprise.grandcentral.SignalDriverAnnotation\22, dir = \22sink\22, id = [[ID]] : i64, peer = \22~Top|Foo>dataIn.a.b.c\22, side = \22local\22, targetId = 6 : i64}>">
; CHECK-SAME: #firrtl<"subAnno<fieldID = 2, {class = \22sifive.enterprise.grandcentral.SignalDriverAnnotation\22, dir = \22sink\22, id = [[ID]] : i64, peer = \22~Top|Foo>dataIn.d\22, side = \22local\22, targetId = 7 : i64}>">
; CHECK-SAME: #firrtl<"subAnno<fieldID = 3, {class = \22sifive.enterprise.grandcentral.SignalDriverAnnotation\22, dir = \22sink\22, id = [[ID]] : i64, peer = \22~Top|Foo>dataIn.e\22, side = \22local\22, targetId = 8 : i64}>">

; // -----

; Grand Central's ModuleReplacementAnnotation is properly scattered to the circuit
; and the targeted operations.

circuit Top : %[[{
  "class": "sifive.enterprise.grandcentral.ModuleReplacementAnnotation",
  "targets": [
    "~Top|Top/child:Child",
    "~Top|Top/childWrapper:ChildWrapper/child:Child"
  ],
  "circuit": "",
  "annotations": ["foo", "bar"],
  "circuitPackage": "other",
  "dontTouches":[
    "~Top|Child>in",
    "~Top|Child>out"
  ]
}]]
  module ChildWrapper :
    input in : UInt<123>
    output out : UInt<456>

    inst child of Child
    child.in <= in
    out <= child.out

  extmodule Child :
    input in : UInt<123>
    output out : UInt<456>

  module Top :
    inst child of Child
    inst childWrapper of ChildWrapper

; CHECK-LABEL: firrtl.circuit "Top"
; CHECK-SAME: {annotations = ["foo", "bar"], circuit = "", circuitPackage = "other", class = "sifive.enterprise.grandcentral.ModuleReplacementAnnotation", id = [[ID:.+]] : i64}

; CHECK: %child_in, %child_out = firrtl.instance child
; CHECK-SAME: {annotations = [{circt.nonlocal = @"~Top|Top/childWrapper:ChildWrapper/child:Child", class = "circt.nonlocal"}]}

; CHECK: firrtl.extmodule @Child(
; CHECK-SAME:   in in: !firrtl.uint<123> [{class = "firrtl.transforms.DontTouchAnnotation"}],
; CHECK-SAME:   out out: !firrtl.uint<456> [{class = "firrtl.transforms.DontTouchAnnotation"}]
; CHECK-SAME: )
; CHECK-SAME: attributes {annotations = [
; CHECK-SAME:   {circt.nonlocal = @"~Top|Top/child:Child", id = [[ID]] : i64},
; CHECK-SAME:   {circt.nonlocal = @"~Top|Top/childWrapper:ChildWrapper/child:Child", id = [[ID]] : i64}
; CHECK-SAME: ]}
