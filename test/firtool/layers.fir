; RUN: firtool %s -split-input-file | FileCheck %s

FIRRTL version 4.0.0
circuit Foo: %[[
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Foo|Foo>x"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Foo|Foo>y"
  }
]]
  layer A, bind:
    layer B, bind:

  module Foo:
    input in: UInt<1>

    layerblock A:
      node x = in

      layerblock B:
        node y = x

; CHECK-LABEL: module Foo_A_B(
; CHECK-NEXT:    input _x
; CHECK-NEXT:  );
; CHECK:         wire y = _x;
; CHECK-NEXT:  endmodule

; CHECK-LABEL: module Foo_A(
; CHECK-NEXT:    input _in
; CHECK:         wire x = _in;
; CHECK-NEXT:    wire x_probe = x;
; CHECK-NEXT:  endmodule

; CHECK-LABEL: FILE "groups_Foo_A_B.sv"
; CHECK:       `include "groups_Foo_A.sv"
; CHECK-NEXT:  `ifndef groups_Foo_A_B
; CHECK-NEXT:  `define groups_Foo_A_B
; CHECK-NEXT:  bind Foo Foo_A_B foo_A_B (
; CHECK-NEXT:    _x (Foo.foo_A.x_probe)
; CHECK-NEXT:  );
; CHECK-NEXT:  `endif // groups_Foo_A_B

; CHECK-LABEL: FILE "groups_Foo_A.sv"
; CHECK:       `ifndef groups_Foo_A
; CHECK-NEXT:  `define groups_Foo_A
; CHECK-NEXT:   bind Foo Foo_A foo_A (
; CHECK-NEXT:     ._in (in)
; CHECK-NEXT:   );
; CHECK-NEXT:  `endif // groups_Foo_A

; // -----

FIRRTL version 4.0.0
circuit Foo: %[[
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Foo|Bar>b"
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Foo|Foo>c"
  }
]]
  layer A, bind:

  module Bar:
    output a: Probe<UInt<1>>
    input x: UInt<1>

    node b = UInt<1>(0)

    layerblock A:
      define a = probe(b)

  module Foo:
    input x: UInt<1>

    inst bar of Bar
    connect bar.x, x

    layerblock A:

      node c = read(bar.a)

; CHECK-LABEL: module Bar_A(
; CHECK-NEXT:    input _b
; CHECK-NEXT:  );
; CHECK:         wire _b_probe = _b;
; CHECK-NEXT:  endmodule

; CHECK-LABEL: module Foo_A(
; CHECK-NEXT:    input _bar_a
; CHECK-NEXT:  );
; CHECK:         wire c = _bar_a;
; CHECK-NEXT:  endmodule

; CHECK-LABEL: FILE "groups_Foo_A.sv"
; CHECK:       `ifndef groups_Foo_A
; CHECK-NEXT:  `define groups_Foo_A
; CHECK-NEXT:  bind Bar Bar_A bar_A (
; CHECK-NEXT:    ._b (b)
; CHECK-NEXT:  );
; CHECK-NEXT:  bind Foo Foo_A foo_A (
; CHECK-NEXT:    ._bar_a (Foo.bar.bar_A._b_probe)
; CHECK-NEXT:  );
; CHECK-NEXT:  `endif // groups_Foo_A
