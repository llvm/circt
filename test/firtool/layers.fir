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
  },
  {
    "class": "firrtl.transforms.DontTouchAnnotation",
    "target": "~Foo|Foo>z"
  }
]]
  layer A, bind:
    layer B, bind:

  public module Foo:
    input in: UInt<1>

    input clock: Clock
    input cond: UInt<1>
    input enable: UInt<1>

    layerblock A:
      node x = in

      layerblock B:
        node y = x

      when cond:
        layerblock B:
          when x:
            node z = x
            assert(clock, cond, enable, "Test")

; CHECK-LABEL: module Foo_A_B(
; CHECK-NEXT:    input x,
; CHECK-NEXT:          cond,
; CHECK-NEXT:          enable,
; CHECK-NEXT:          clock
; CHECK-NEXT:  );
; CHECK:         wire y = x;
; CHECK:         wire z = x;
; CHECK:         always @(posedge clock) begin
; CHECK-NEXT:      if (cond & x & enable)
; CHECK-NEXT:        assert(cond) else $error("Test");
; CHECK-NEXT:    end // always @(posedge)
; CHECK-NEXT:  endmodule

; CHECK-LABEL: module Foo_A(
; CHECK-NEXT:    input in
; CHECK:         wire x = in;
; CHECK-NEXT:    wire x_probe = x;
; CHECK-NEXT:  endmodule

; CHECK-LABEL: FILE "layers_Foo_A_B.sv"
; CHECK:       `include "layers_Foo_A.sv"
; CHECK-NEXT:  `ifndef layers_Foo_A_B
; CHECK-NEXT:  `define layers_Foo_A_B
; CHECK-NEXT:  bind Foo Foo_A_B a_b (
; CHECK-NEXT:    .x (Foo.a.x_probe),
; CHECK-NEXT:    .cond (cond),
; CHECK-NEXT:    .enable (enable),
; CHECK-NEXT:    .clock (clock)
; CHECK-NEXT:  );
; CHECK-NEXT:  `endif // layers_Foo_A_B

; CHECK-LABEL: FILE "layers_Foo_A.sv"
; CHECK:       `ifndef layers_Foo_A
; CHECK-NEXT:  `define layers_Foo_A
; CHECK-NEXT:   bind Foo Foo_A a (
; CHECK-NEXT:     .in (in)
; CHECK-NEXT:   );
; CHECK-NEXT:  `endif // layers_Foo_A
