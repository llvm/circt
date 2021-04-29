; RUN: circt-translate -import-firrtl -split-input-file %s | FileCheck %s

; COM: An Annotation without a target is promoted to a CircuitTarget
; COM: Annotation. This represents a Scala FIRRTL Compiler
; COM: `firrtl.annotations.NoTargetAnnotation`.
circuit Foo: %[[{"a":"a"}]]
  module Foo:
    skip

    ; CHECK-LABEL: module {
    ; CHECK: firrtl.circuit "Foo" attributes {annotations = [{a = "a"}]}

; // -----

; COM: A legacy `firrtl.annotations.CircuitName` annotation becomes a
; COM: CircuitTarget Annotation.
circuit Foo: %[[{"a":"a","target":"Foo"}]]
  module Foo:
    skip

    ; CHECK-LABEL: module {
    ; CHECK: firrtl.circuit "Foo" attributes {annotations = [{a = "a"}]}

; // -----

; COM: A CircuitTarget Annotation is attached to the circuit.
circuit Foo: %[[{"a":"a","target":"~Foo"}]]
  module Foo:
    skip

    ; CHECK-LABEL: module {
    ; CHECK: firrtl.circuit "Foo" attributes {annotations = [{a = "a"}]}

; // -----

; COM: A legacy `firrtl.annotations.ModuleName` annotation becomes a
; COM: ModuleTarget Annotation
circuit Foo: %[[{"a":"a","target":"Foo.Foo"}]]
  module Foo:
    skip

    ; CHECK-LABEL: module {
    ; CHECK: firrtl.module @Foo() attributes {annotations = [{a = "a"}]}

; // -----

; COM: A ModuleTarget Annotation is attached to the correct module.
circuit Foo: %[[{"a":"a","target":"~Foo|Foo"}]]
  module Foo:
    skip

    ; CHECK-LABEL: module {
    ; CHECK: firrtl.module @Foo() attributes {annotations = [{a = "a"}]}

; // -----

; COM: A ModuleTarget Annotation can be attached to an ExtModule.
circuit Foo: %[[{"a":"a","target":"~Foo|Bar"}]]
  extmodule Bar:
    input a: UInt<1>
  module Foo:
    input a: UInt<1>
    inst bar of Bar
    bar.a <= a

    ; CHECK-LABEL: module {
    ; CHECK: firrtl.extmodule @Bar
    ; CHECK-SAME: attributes {annotations = [{a = "a"}]}

; // -----

; COM: A ReferenceTarget/ComponentName pointing at an Instance should work.
circuit Foo: %[[{"a":"a","target":"~Foo|Foo>bar"},{"b":"b","target":"Foo.Foo.bar"}]]
  module Bar:
    skip
  module Foo:
    inst bar of Bar

    ; CHECK-LABEL: module {
    ; CHECK: firrtl.instance @Bar
    ; CHECK-SAME: annotations = [{a = "a"}, {b = "b"}]

; // -----

; COM: A ReferenceTarget/ComponentName pointing at a CMem should work.
circuit Foo: %[[{"a":"a","target":"~Foo|Foo>bar"},{"b":"b","target":"Foo.Foo.bar"}]]
  module Foo:
    cmem bar: UInt<1>[8]

    ; CHECK-LABEL: module {
    ; CHECK: firrtl.cmem
    ; CHECK-SAME: annotations = [{a = "a"}, {b = "b"}]

; // -----

; COM: A ReferenceTarget/ComponentName pointing at a memory should work.
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

; COM: A ReferenceTarget/ComponentName pointing at a node should work.
; COM: This shouldn't crash if the node is in a nested block.
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

; COM: A ReferenceTarget/ComponentName pointing at a wire should work.
circuit Foo: %[[{"a":"a","target":"~Foo|Foo>bar"},{"b":"b","target":"Foo.Foo.bar"}]]
  module Foo:
    wire bar: UInt<1>

    ; CHECK-LABEL: module {
    ; CHECK: %bar = firrtl.wire
    ; CHECK-SAME: annotations = [{a = "a"}, {b = "b"}]

; // -----

; COM: A ReferenceTarget/ComponentName pointing at a register should work.
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

; COM: A ReferenceTarget/ComponentName pointing at an SMem should work.
circuit Foo: %[[{"a":"a","target":"~Foo|Foo>bar"},{"b":"b","target":"Foo.Foo.bar"}]]
  module Foo:
    smem bar: UInt<1>[8]

    ; CHECK-LABEL: module {
    ; CHECK: firrtl.smem
    ; CHECK-SAME: annotations = [{a = "a"}, {b = "b"}]

; // -----

; COM: A ReferenceTarget/ComponentName pointing at a module/extmodule port should work.
circuit Foo: %[[{"a":"a","target":"~Foo|Bar>bar"},{"b":"b","target":"Foo.Foo.foo"}]]
  extmodule Bar:
    input bar: UInt<1>
  module Foo:
    input foo: UInt<1>
    inst bar of Bar
    bar.bar <= foo

    ; CHECK-LABEL: module {
    ; CHECK: firrtl.extmodule @Bar
    ; CHECK-SAME: [[_:.+]] {firrtl.annotations = [{a = "a"}]}
    ; CHECK: firrtl.module @Foo
    ; CHECK-SAME: %foo: [[_:.+]] {firrtl.annotations = [{b = "b"}]}

; // -----

; COM: All types of JSON values should work
circuit Foo: %[[{"string":"a","integer":42,"float":3.14,"boolean":true,"null":null,"object":{"foo":"bar"},"array":[1,2,3]}]]
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

; COM: JSON escapes should work.
circuit Foo: %[[{"\"":"}]]"}]]
  module Foo:
    skip

    ; CHECK-LABEL: module {
    ; CHECK: firrtl.circuit "Foo" attributes {annotations =

; // -----

; COM: Subfield/Subindex annotations should be parsed correctly on wires
circuit Foo: %[[{"one":null,"target":"~Foo|Foo>bar.qux"},{"two":null,"target":"~Foo|Foo>bar[1].baz"} ]]
  module Foo:
    wire bar: {baz: UInt<1>, qux: UInt<1>}[2]

    ; CHECK-LABEL: module {
    ; CHECK: %bar = firrtl.wire  {annotations = [{one, target = [".qux"]}, {target = ["[1]", ".baz"], two}]} : !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>


; // -----

; COM: Subfield/Subindex annotations should be parsed correctly on registers
circuit Foo: %[[{"one":null,"target":"~Foo|Foo>bar.qux"},{"two":null,"target":"~Foo|Foo>bar[1].baz"} ]]
  module Foo:
    input clock: Clock
    reg bar: {baz: UInt<1>, qux: UInt<1>}[2], clock

    ; CHECK-LABEL: module {
    ; CHECK: %bar = firrtl.reg %clock  {annotations = [{one, target = [".qux"]}, {target = ["[1]", ".baz"], two}]} : (!firrtl.clock) -> !firrtl.vector<bundle<baz: uint<1>, qux: uint<1>>, 2>

; // -----

; COM: Annotations should apply even when the target's name is dropped.
circuit Foo: %[[{"target": "Foo|Foo>_T_0", "a": "a"},
                {"target": "Foo|Foo>_T_1", "a": "a"},
                {"target": "Foo|Foo>_T_2", "a": "a"},
                {"target": "Foo|Foo>_T_3", "a": "a"},
                {"target": "Foo|Foo>_T_4", "a": "a"},
                {"target": "Foo|Foo>_T_5", "a": "a"},
                {"target": "Foo|Foo>_T_6", "a": "a"},
                {"target": "Foo|Foo>_T_7", "a": "a"},
                {"target": "Foo|Foo>_T_8", "a": "a"}]]
  module Bar:
    skip
  module Foo:
    input reset : UInt<1>
    input clock : Clock

    ; CHECK: %0 = firrtl.wire  {annotations = [{a = "a"}]}
    wire _T_0 : UInt<1>
    ; CHECK-NOT: firrtl.node
    node _T_1 = _T_0
    ; CHECK: %1 = firrtl.reg %clock  {annotations = [{a = "a"}]}
    reg _T_2 : UInt<1>, clock
    ; CHECK: %2 = firrtl.regreset %clock, %reset, %c0_ui4  {annotations = [{a = "a"}]}
    reg _T_3 : UInt<4>, clock with :
      reset => (reset, UInt<4>("h0"))
    ; CHECK: %3 = firrtl.smem Undefined  {annotations = [{a = "a"}], name = ""}
    smem _T_4 : UInt<1>[9] [256]
    ; CHECK: %4 = firrtl.cmem  {annotations = [{a = "a"}], name = ""}
    cmem _T_5 : UInt<1>[9] [256]
    ; CHECK: %5 = firrtl.memoryport Infer %4, %reset, %clock {annotations = [{a = "a"}], name = ""}
    infer mport _T_6 = _T_5[reset], clock
    ; CHECK: firrtl.instance @Bar  {annotations = [{a = "a"}], name = ""}
    inst _T_7 of Bar
    ; CHECK: firrtl.mem Undefined  {annotations = [{a = "a"}], depth = 8 : i64, name = "", portNames = ["w"], readLatency = 0 : i32, writeLatency = 1 : i32}
    mem _T_8 :
        data-type => UInt<4>
        depth => 8
        writer => w
        read-latency => 0
        write-latency => 1
        read-under-write => undefined

; // -----

; COM: DontTouch annotation preserves temporary names
circuit Foo: %[[{"target": "Foo|Foo>_T_0", "class": "firrtl.transforms.DontTouchAnnotation"},
                {"target": "Foo|Foo>_T_1", "class": "firrtl.transforms.DontTouchAnnotation"},
                {"target": "Foo|Foo>_T_2", "class": "firrtl.transforms.DontTouchAnnotation"},
                {"target": "Foo|Foo>_T_3", "class": "firrtl.transforms.DontTouchAnnotation"},
                {"target": "Foo|Foo>_T_4", "class": "firrtl.transforms.DontTouchAnnotation"},
                {"target": "Foo|Foo>_T_5", "class": "firrtl.transforms.DontTouchAnnotation"},
                {"target": "Foo|Foo>_T_6", "class": "firrtl.transforms.DontTouchAnnotation"},
                {"target": "Foo|Foo>_T_7", "class": "firrtl.transforms.DontTouchAnnotation"},
                {"target": "Foo|Foo>_T_8", "class": "firrtl.transforms.DontTouchAnnotation"}]]
  module Bar:
    skip
  module Foo:
    input reset : UInt<1>
    input clock : Clock

    ; CHECK: %_T_0 = firrtl.wire  {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]}
    wire _T_0 : UInt<1>
    ; CHECK: %_T_1 = firrtl.node %_T_0  {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]}
    node _T_1 = _T_0
    ; CHECK: %_T_2 = firrtl.reg %clock  {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]}
    reg _T_2 : UInt<1>, clock
    ; CHECK: %_T_3 = firrtl.regreset %clock, %reset, %c0_ui4  {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]}
    reg _T_3 : UInt<4>, clock with :
      reset => (reset, UInt<4>("h0"))
    ; CHECK: %_T_4 = firrtl.smem Undefined  {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}], name = "_T_4"}
    smem _T_4 : UInt<1>[9] [256]
    ; CHECK: %_T_5 = firrtl.cmem  {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}], name = "_T_5"}
    cmem _T_5 : UInt<1>[9] [256]
    ; CHECK: %_T_6 = firrtl.memoryport Infer %_T_5, %reset, %clock {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}], name = "_T_6"}
    infer mport _T_6 = _T_5[reset], clock
    ; CHECK: firrtl.instance @Bar  {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}], name = "_T_7"}
    inst _T_7 of Bar
    ; CHECK: firrtl.mem Undefined  {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}], depth = 8 : i64, name = "_T_8", portNames = ["w"], readLatency = 0 : i32, writeLatency = 1 : i32}
    mem _T_8 :
        data-type => UInt<4>
        depth => 8
        writer => w
        read-latency => 0
        write-latency => 1
        read-under-write => undefined
