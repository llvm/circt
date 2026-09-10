; RUN: firtool --disable-all-randomization %s | FileCheck %s

; Check that the probes are correctly sunk into the layer, and there are
; no bare probes left in the design.

; Check the probes are inlined into the assertions:

; CHECK-LABEL: module Foo_A();
; CHECK-NEXT:    always @(posedge Foo.clock) begin
; CHECK-NEXT:      assert(Foo.bar.`ref_Bar_probe) else $error("message");
; CHECK-NEXT:      assert(Foo.bar.`ref_Bar_probe) else $error("message");
; CHECK-NEXT:    end // always @(posedge)
; CHECK-NEXT:    `ifndef SYNTHESIS
; CHECK-NEXT:       initial
; CHECK-NEXT:         force Foo.bar.`ref_Bar_rwprobe = 1'h1;
; CHECK-NEXT:    `endif // not def SYNTHESIS
; CHECK-NEXT:  endmodule

; Check that there are no probes in the top level module:

; CHECK-LABEL: module Foo(
; CHECK-NEXT:    input clock
; CHECK-NEXT:  );
; CHECK-EMPTY:
; CHECK-NEXT:    Bar bar ();
; CHECK-NEXT:  endmodule

FIRRTL version 4.0.0

circuit Foo:
  layer A, bind:

  public module Foo:
    input clock: Clock

    inst bar of Bar
    node cond = read(bar.probe)

    layerblock A:
      assert(clock, cond, UInt<1>(1), "message")
      assert(clock, cond, UInt<1>(1), "message")
      force_initial(bar.rwprobe, UInt<1>(1))

  extmodule Bar:
    output probe : Probe<UInt<1>>
    output rwprobe : RWProbe<UInt<1>>
