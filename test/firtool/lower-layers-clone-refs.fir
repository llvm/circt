; RUN: firtool --disable-all-randomization --advanced-layer-sink %s | FileCheck %s

; Check that there is no probe wire in the output example.

; CHECK:      module Example_A();
; CHECK-NEXT:   wire _GEN = 1'h0;
; CHECK-NEXT: endmodule

; CHECK:      module Example_B();
; CHECK-NEXT:   wire _GEN = 1'h0;
; CHECK-NEXT: endmodule

; CHECK:      module Example();
; CHECK-NEXT: endmodule

FIRRTL version 5.0.0

circuit Example:
  layer A, bind:
  layer B, bind:

  public module Example:
    output pa: Probe<UInt<1>, A>
    output pb: Probe<UInt<1>, B>

    node n = UInt<1>(0)
    layerblock A:
      define pa = probe(n)
    layerblock B:
      define pb = probe(n)
