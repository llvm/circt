; RUN: firtool %s | FileCheck %s

FIRRTL version 4.0.0

; Check use of colored rwprobe in enablelayers.
; Check rwprobe of hardware in a layer.

; Order isn't critical, but for simplicity just check lines.

;      CHECK: module TH();
; CHECK-NEXT:   `ifndef SYNTHESIS
; CHECK-NEXT:     initial
; CHECK-NEXT:       force TH.d.t.w = 1'h1;
; CHECK-NEXT:   `endif // not def SYNTHESIS
; CHECK-NEXT:   DUT d ();
; CHECK-NEXT: endmodule

;      CHECK: module DUT_T();
; CHECK-NEXT:   wire w = 1'h0;
; CHECK-NEXT: endmodule
;
;      CHECK: module DUT();
; CHECK-NEXT: endmodule

circuit TH:
  layer T, bind:

  public module TH enablelayer T:
    inst d of DUT
    force_initial(d.p, UInt<1>(1))

  public module DUT:
    output p : RWProbe<UInt<1>, T>

    layerblock T:
      wire w : UInt<1>
      connect w, UInt<1>(0)
      define p = rwprobe(w)

