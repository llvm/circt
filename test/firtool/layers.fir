; RUN: firtool %s | FileCheck %s

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
