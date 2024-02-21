; RUN: firtool -verilog %s | FileCheck %s

; This is an end-to-end example of a test-bench (Foo) enabling verification,
; probing into a device-under-test (Bar), and reading from hardware which is
; only present if the verification layer is enabled.

FIRRTL version 4.0.0

circuit Foo: %[[
  {"class": "firrtl.transforms.DontTouchAnnotation", "target": "~Foo|Bar>c"},
  {"class": "firrtl.transforms.DontTouchAnnotation", "target": "~Foo|Foo>d"}
]]
  layer Verification, bind:

  ; CHECK: module Bar_Verification();
  ; CHECK:   wire c = 1'h0;
  ; CHECK:   wire c_probe = c;
  ; CHECK: endmodule

  ; CHECK: module Bar();
  ; CHECK: endmodule
  module Bar:
    input a: UInt<1>
    output b: Probe<UInt<1>, Verification>

    layerblock Verification:
      node c = UInt<1>(0)
      define b = probe(c)

  ; CHECK: module Foo();
  ; CHECK:   wire d = Foo.bar.verification.c_probe;
  ; CHECK:   Bar bar ();
  ; CHECK: endmodule
  public module Foo enablelayer Verification:

    inst bar of Bar

    node d = read(bar.b)
    connect bar.a, d
