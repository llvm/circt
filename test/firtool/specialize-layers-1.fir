; RUN: firtool %s -disable-layers=A -split-input-file | FileCheck %s

; Check that layer specialization works.
; CHECK-LABEL: module LayerSpecialization();
; CHECK-NEXT:  endmodule
FIRRTL version 4.0.0
circuit LayerSpecialization:
  layer A, bind:
  public module LayerSpecialization:
    layerblock A:
      wire w : UInt<1>
      invalidate w
