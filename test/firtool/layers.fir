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

; CHECK-LABEL: module Foo_A_B();
; CHECK:         wire y = Foo.a.x_0;
; CHECK:         wire z = Foo.a.x_0;
; CHECK:         always @(posedge Foo.clock) begin
; CHECK-NEXT:      if (Foo.cond & Foo.a.x_0 & Foo.enable)
; CHECK-NEXT:        assert(Foo.cond) else $error("Test");
; CHECK-NEXT:    end // always @(posedge)
; CHECK-NEXT:  endmodule

; CHECK-LABEL: module Foo_A();
; CHECK-NEXT:    wire x = Foo.in;
; CHECK-NEXT:    wire x_0 = x;
; CHECK-NEXT:  endmodule

; CHECK-LABEL: FILE "layers-Foo-A-B.sv"
; CHECK:       `include "layers-Foo-A.sv"
; CHECK-NEXT:  `ifndef layers_Foo_A_B
; CHECK-NEXT:  `define layers_Foo_A_B
; CHECK-NEXT:  bind Foo Foo_A_B a_b ();
; CHECK-NEXT:  `endif // layers_Foo_A_B

; CHECK-LABEL: FILE "layers-Foo-A.sv"
; CHECK:       `ifndef layers_Foo_A
; CHECK-NEXT:  `define layers_Foo_A
; CHECK-NEXT:   bind Foo Foo_A a ();
; CHECK-NEXT:  `endif // layers_Foo_A
