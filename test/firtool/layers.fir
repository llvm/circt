; RUN: firtool -split-input-file %s | FileCheck %s

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
; CHECK:         wire y = Foo.a.x;
; CHECK:         wire z = Foo.a.x;
; CHECK:         always @(posedge Foo.clock) begin
; CHECK-NEXT:      if (Foo.cond & Foo.a.x & Foo.enable)
; CHECK-NEXT:        assert(Foo.cond) else $error("Test");
; CHECK-NEXT:    end // always @(posedge)
; CHECK-NEXT:  endmodule

; CHECK-LABEL: module Foo_A();
; CHECK-NEXT:    wire x = Foo.in;
; CHECK-NEXT:  endmodule

; CHECK-LABEL: FILE "layers-Foo-A.sv"
; CHECK:       `ifndef layers_Foo_A
; CHECK-NEXT:    `define layers_Foo_A
; CHECK-NEXT:     bind Foo Foo_A a ();
; CHECK-NEXT:  `endif // not def layers_Foo_A

; CHECK-LABEL: FILE "layers-Foo-A-B.sv"
; CHECK:       `ifndef layers_Foo_A_B
; CHECK-NEXT:    `define layers_Foo_A_B
; CHECK-NEXT:    `include "layers-Foo-A.sv"
; CHECK-NEXT:    bind Foo Foo_A_B a_b ();
; CHECK-NEXT:  `endif // not def layers_Foo_A_B

; // -----

FIRRTL version 5.0.0

circuit Foo:
  layer A, inline:
    layer B, bind:

  ; Check contents of bound-in-module and bindfile.

  ; CHECK: module Foo_A_B();
  ; CHECK:   always @(posedge Foo.clock) begin
  ; CHECK:     if (Foo.enable)
  ; CHECK:       assert(Foo.cond) else $error("message");
  ; CHECK: endmodule

  ; CHECK: FILE "layers-Foo-A-B.sv"
  ; CHECK: `ifndef layers_Foo_A_B
  ; CHECK:   `define layers_Foo_A_B
  ; CHECK:   `ifndef layer$A
  ; CHECK:     `_ERROR_A_not_enabled
  ; CHECK:   `endif // not def layer$A
  ; CHECK:   bind Foo Foo_A_B a_b ();
  ; CHECK: `endif // not def layers_Foo_A_B

  public module Foo:
    input clock : Clock
    input cond : UInt<1>
    input enable : UInt<1>

    layerblock A:
      layerblock B:
        assert(clock, cond, enable, "message")
