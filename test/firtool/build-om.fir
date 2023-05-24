; RUN: firtool %s -ir-fir | FileCheck %s

circuit Top : %[[
  {
    "class": "circt.experimental.ConstPropertyAnnotation",
    "target": "~Top|Top>a"
  },
  {
    "class": "circt.experimental.ConstPropertyAnnotation",
    "target": "~Top|Top>b"
  },
  {
    "class": "circt.experimental.ConstPropertyAnnotation",
    "target": "~Top|Top>c"
  },
  {
    "class": "circt.experimental.ConstPropertyAnnotation",
    "target": "~Top|Foo>x"
  },
  {
    "class": "circt.experimental.ConstPropertyAnnotation",
    "target": "~Top|Foo>y"
  }
]]
  module Top:
    input a: const SInt<64>
    output b: const SInt<64>
    output c: const SInt<64>
    output d: const SInt<64>
    b <= a
    inst foo of Foo
    foo.x <= a
    c <= foo.y
    d <= SInt<64>(5)

  module Foo:
    input x: const SInt<64>
    output y: const SInt<64>
    y <= x

; CHECK-LABEL: om.class @Top(%a: !firrtl.const.sint<64>) {
; CHECK-NEXT:    %0 = om.object @Foo(%a) : (!firrtl.const.sint<64>) -> !om.class.type<@Foo>
; CHECK-NEXT:    %1 = om.object.field %0, [@y] : (!om.class.type<@Foo>) -> !firrtl.const.sint<64>
; CHECK-NEXT:    om.class.field @b, %a : !firrtl.const.sint<64>
; CHECK-NEXT:    om.class.field @c, %1 : !firrtl.const.sint<64>
; CHECK-NEXT:  }

; CHECK-LABEL: om.class @Foo(%x: !firrtl.const.sint<64>) {
; CHECK-NEXT:    om.class.field @y, %x : !firrtl.const.sint<64>
; CHECK-NEXT:  }
