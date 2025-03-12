; RUN: firtool %s -ir-verilog | FileCheck %s

FIRRTL version 4.3.0

circuit Test : %[[
{
  "class": "firrtl.transforms.MustDeduplicateAnnotation",
  "modules": ["~Test|CPU_1", "~Test|CPU_2"]
}
]]
  ; CHECK: hw.hierpath private [[NLA1:@.+]] [@CPU_1::[[SYM1:@.+]]]
  ; CHECK: hw.hierpath private [[NLA2:@.+]] [@CPU_1::[[SYM2:@.+]], @Fetch_1::[[SYM3:@.+]]]
  public module Test :
    input in : UInt<1>
    output out_1 : UInt<1>
    output out_2 : UInt<1>
    output om_out_1 : AnyRef
    output om_out_2 : AnyRef
    inst cpu_1 of CPU_1
    inst cpu_2 of CPU_2
    connect cpu_1.in, in
    connect cpu_2.in, in
    connect out_1, cpu_1.out
    connect out_2, cpu_2.out
    propassign om_out_1, cpu_1.om_out
    propassign om_out_2, cpu_2.om_out

  ; CHECK-LABEL: hw.module private @CPU_1
  ; CHECK-SAME: out out : i1 {hw.exportPort = #hw<innerSym[[SYM1]]>}
  module CPU_1 :
    input in : UInt<1>
    output out : UInt<1>
    output om_out : AnyRef

    object om of OM_1
    propassign om_out, om

    ; CHECK: hw.instance "fetch_1" sym [[SYM2]]
    inst fetch_1 of Fetch_1
    inst fetch_2 of Fetch_1
    connect fetch_1.in, in
    connect fetch_2.in, in
    connect out, fetch_1.out

  ; CHECK-NOT: CPU_2
  module CPU_2 :
    input in : UInt<1>
    output out : UInt<1>
    output om_out : AnyRef

    object om of OM_2
    propassign om_out, om

    inst fetch_1 of Fetch_2
    inst fetch_2 of Fetch_2
    connect fetch_1.in, in
    connect fetch_2.in, in
    connect out, fetch_1.out

  module Fetch_1 :
    input in : UInt<1>
    output out : UInt<1>
    ; CHECK: %foo = sv.wire sym [[SYM3]]
    wire foo : UInt<1>
    connect foo, in
    connect out, foo

  ; CHECK-NOT: Fetch_2
  module Fetch_2 :
    input in : UInt<1>
    output out : UInt<1>
    wire foo : UInt<1>
    connect foo, in
    connect out, foo

  class Foo_1 :
    output out_foo : Integer
    propassign out_foo, Integer(1)

  class Foo_2 :
    output out_bar : Integer
    propassign out_bar, Integer(1)

  class Foo_3 :
    output out_baz : Integer
    propassign out_baz, Integer(1)

  class Foo_4 :
    output out_baz : Integer
    propassign out_baz, Integer(1)

  ; CHECK-LABEL: om.class @OM_1(%basepath: !om.basepath) -> (out_foo_1: !om.class.type<@Foo_1>, out_foo_2: !om.class.type<@Foo_2>, out_foo_3: !om.class.type<@Foo_3>, out_foo_4: !om.class.type<@Foo_3>, out_1: !om.path, out_2: !om.path)
  class OM_1 :
    output out_1 : Path
    output out_2 : Path
    output out_foo_1 : Inst<Foo_1>
    output out_foo_2 : Inst<Foo_2>
    output out_foo_3 : Inst<Foo_3>
    output out_foo_4 : Inst<Foo_4>

    ; CHECK: [[FOO_1:%.+]] = om.object @Foo_1
    object foo_1 of Foo_1
    propassign out_foo_1, foo_1

    ; CHECK: [[FOO_2:%.+]] = om.object @Foo_2
    object foo_2 of Foo_2
    propassign out_foo_2, foo_2

    ; CHECK: [[FOO_3:%.+]] = om.object @Foo_3
    object foo_3 of Foo_3
    propassign out_foo_3, foo_3

    ; CHECK: [[FOO_4:%.+]] = om.object @Foo_3
    object foo_4 of Foo_4
    propassign out_foo_4, foo_4

    ; CHECK: om.path_create reference %basepath [[NLA1]]
    propassign out_1, path("OMReferenceTarget:~Test|CPU_1>out")
    ; CHECK: om.path_create reference %basepath [[NLA2]]
    propassign out_2, path("OMReferenceTarget:~Test|CPU_1/fetch_1:Fetch_1>foo")

    ; CHECK: om.class.fields [[FOO_1]], [[FOO_2]], [[FOO_3]], [[FOO_4]], %4, %5 : !om.class.type<@Foo_1>, !om.class.type<@Foo_2>, !om.class.type<@Foo_3>, !om.class.type<@Foo_3>, !om.path, !om.path

  ; CHECK-NOT: OM_2
  class OM_2 :
    output out_1 : Path
    output out_2 : Path
    output out_foo_1 : Inst<Foo_1>
    output out_foo_2 : Inst<Foo_2>
    output out_foo_3 : Inst<Foo_3>
    output out_foo_4 : Inst<Foo_4>

    object foo_1 of Foo_1
    propassign out_foo_1, foo_1

    object foo_2 of Foo_2
    propassign out_foo_2, foo_2

    object foo_3 of Foo_3
    propassign out_foo_3, foo_3

    object foo_4 of Foo_4
    propassign out_foo_4, foo_4

    propassign out_1, path("OMReferenceTarget:~Test|CPU_2>out")
    propassign out_2, path("OMReferenceTarget:~Test|CPU_2/fetch_1:Fetch_2>foo")
