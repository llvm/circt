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
    ; CHECK: firrtl.extmodule private @Bar
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
    ; CHECK: firrtl.hierpath  @nla_1 [@Foo::@bar, @Bar]
    ; CHECK: firrtl.module private @Bar
    ; CHECK-SAME annotations = [{c = "c"}]
    ; CHECK: firrtl.module @Foo
    ; CHECK: firrtl.instance bar
    ; CHECK-SAME: annotations = [{a = "a"}, {b = "b"}]

; // -----

; Test result annotations of InstanceOp.
; Must add inner_sym, if any subfield of a bundle type has nonlocal anchor.
; Otherwise, the nla will be illegal, without any inner_sym.
; Test on port and wire.
circuit Foo: %[[{"one":null,"target":"~Foo|Foo>bar.a"},
                {"two":null,"target":"~Foo|Foo>bar.b.baz"},
                {"three":null,"target":"~Foo|Foo/bar:Bar>b.qux"},
                {"five":null,"target":"~Foo|Foo/bar:Bar>d.qux"},
                {"four":null,"target":"Foo.Foo.bar.c"}

                ]]
  module Bar:
    input a: UInt<1>
    output b: {baz: UInt<1>, qux: UInt<1>}
    output c: UInt<1>
    wire d: {baz: UInt<1>, qux: UInt<1>}
  module Foo:
    inst bar of Bar

    ; CHECK-LABEL: firrtl.circuit "Foo"
    ; CHECK: firrtl.hierpath @nla_2 [@Foo::@bar, @Bar::@d]
    ; CHECK: firrtl.hierpath @nla_1 [@Foo::@bar, @Bar::@b]
    ; CHECK: firrtl.module private @Bar
    ; CHECK-SAME: sym @b [#firrtl.subAnno<fieldID = 2, {circt.nonlocal = @nla_1, three}>]
    ; CHECK-NEXT:  %d = firrtl.wire sym @d
    ; CHECK-SAME: {annotations = [#firrtl.subAnno<fieldID = 2, {circt.nonlocal = @nla_2, five}>]}
    ; CHECK-SAME: : !firrtl.bundle<baz: uint<1>, qux: uint<1>>
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
    ; CHECK: chirrtl.combmem
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
    ; CHECK: chirrtl.seqmem
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
    ; CHECK: firrtl.extmodule private @Bar
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

    ; CHECK: %_T_0 = firrtl.wire droppable_name  {annotations = [{a = "a"}]}
    wire _T_0 : UInt<1>
    ; CHECK: %_T_1 = firrtl.node
    node _T_1 = _T_0
    ; CHECK: %_T_2 = firrtl.reg droppable_name %clock  {annotations = [{a = "a"}]}
    reg _T_2 : UInt<1>, clock
    ; CHECK: %_T_3 = firrtl.regreset droppable_name {{.+}}  {annotations = [{a = "a"}]}
    reg _T_3 : UInt<4>, clock with :
      reset => (reset, UInt<4>("h0"))
    ; CHECK: %_T_4 = chirrtl.seqmem droppable_name Undefined {annotations = [{a = "a"}]}
    smem _T_4 : UInt<1>[9] [256]
    ; CHECK: %_T_5 = chirrtl.combmem droppable_name {annotations = [{a = "a"}]}
    cmem _T_5 : UInt<1>[9] [256]
    ; CHECK: chirrtl.memoryport {{.+}} {annotations = [{a = "a"}]
    infer mport _T_6 = _T_5[reset], clock
    ; CHECK: firrtl.instance _T_7 droppable_name {annotations = [{a = "a"}]}
    inst _T_7 of Bar
    ; CHECK: firrtl.mem droppable_name Undefined  {annotations = [{a = "a"}]
    mem _T_8 :
        data-type => UInt<4>
        depth => 8
        writer => w
        read-latency => 0
        write-latency => 1
        read-under-write => undefined

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

; A module with an instance in its body which has the same name as the module
; itself should not cause issues attaching annotations.
; https://github.com/llvm/circt/issues/2709
circuit Test : %[[
  {
    "class":"fake",
    "target":"~Test|Test/Test:Example"
  }
]]
  module Example :
  module Test :
    inst Test of Example

; CHECK-LABEL:  firrtl.circuit "Test"
; CHECK: firrtl.hierpath @nla_1 [@Test::@Test, @Example]
; CHECK: firrtl.module private @Example() attributes {
; CHECK-SAME: annotations = [{circt.nonlocal = @nla_1, class = "fake"}]
; CHECK: firrtl.module @Test()
; CHECK:   firrtl.instance Test sym @Test @Example()

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
; CHECK: firrtl.hierpath @nla_2 [@Foo::@bar, @Bar::@baz, @Baz]
; CHECK: firrtl.hierpath @nla_1 [@Foo::@bar, @Bar::@baz, @Baz]
; CHECK: firrtl.module private @Baz
; CHECK-SAME: annotations = [{a = "a", circt.nonlocal = @nla_1}, {b = "b", circt.nonlocal = @nla_2}]
; CHECK: firrtl.module private @Bar()
; CHECK: firrtl.instance baz sym @baz @Baz()
; CHECK: firrtl.module @Foo()
; CHECK: firrtl.instance bar sym @bar @Bar()

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
; CHECK-SAME: {annotations = [], circuit = "", circuitPackage = "other", class = "sifive.enterprise.grandcentral.SignalDriverAnnotation", id = [[ID:[0-9]+]] : i64, isSubCircuit = true}

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

; CHECK: firrtl.extmodule private @Child(
; CHECK-SAME:   in in: !firrtl.uint<123> sym @in,
; CHECK-SAME:   out out: !firrtl.uint<456> sym @out
; CHECK-SAME: )
; CHECK-SAME: attributes {annotations = [
; CHECK-SAME:   {circt.nonlocal = @nla_1, id = [[ID]] : i64},
; CHECK-SAME:   {circt.nonlocal = @nla_2, id = [[ID]] : i64}
; CHECK-SAME: ]}

; // -----
;
; A numeric "class" shouldn't crash the parser.

circuit Top : %[[
  {
    "class":"0",
    "target":"~Top|Top>a"
  }
]]
  module Top :
    wire a: UInt<1>
    a is invalid

; CHECK-LABEL: firrtl.circuit


; // -----
;
circuit memportAnno: %[[
{
 "class":"test",
 "target":"~memportAnno|memportAnno/foo:Foo>memory.w"
}
]]
  module memportAnno:
    inst foo of Foo
  module Foo:
    mem memory:
      data-type => UInt<8>
      depth => 16
      writer => w
      read-latency => 1
      write-latency => 1
      read-under-write => undefined
; CHECK-LABEL: firrtl.circuit "memportAnno"  {
; CHECK:        firrtl.hierpath @nla_1 [@memportAnno::@foo, @Foo::@memory]
; CHECK:        %memory_w = firrtl.mem sym @memory Undefined  {depth = 16 : i64, name = "memory", portAnnotations
; CHECK-SAME:   [{circt.nonlocal = @nla_1, class = "test"}]

; // -----
; Test that a circuit with annotations that are supposed to be bypassed by
; parsing/scattering are properly moved onto the circuit under a
; "rawAnnotations" attribute.
circuit RawAnnotations: %[[
  {
    "class":"circt.test"
  },
  {
    "class":"circt.testNT"
  },
  {
    "class":"circt.missing"
  }
]]
  module RawAnnotations:
    wire x: UInt<1>

    ; CHECK-LABEL: firrtl.circuit "RawAnnotations"
    ; CHECK-SAME:    rawAnnotations = [
    ; CHECK-SAME:      "circt.test"
    ; Check-SAME:      "circt.testNT"
    ; Check-SAME:      "circt.missing"
