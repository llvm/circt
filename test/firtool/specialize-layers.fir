; RUN: firtool %s | FileCheck %s --check-prefixes=DEFAULT,COMMON
; RUN: firtool %s --disable-layers=A | FileCheck %s --check-prefixes=SPECIALIZE,COMMON
; RUN: firtool %s --enable-layers=A | FileCheck %s --check-prefixes=SPECIALIZE,ENABLE,COMMON

; Check that layer specialization works.

; SPECIALIZE-NOT: LayerSpecialization_A
; DEFAULT: module LayerSpecialization_A

; COMMON-LABEL: module LayerSpecialization();
; ENABLE-NEXT:    Ext ext ();
; COMMON-NEXT:  endmodule

; SPECIALIZE-NOT: LayerSpecialization_A
; DEFAULT: "layers_LayerSpecialization_A.sv"
FIRRTL version 4.0.0
circuit LayerSpecialization:
  layer A, bind:
  extmodule Ext:
  public module LayerSpecialization:
    layerblock A:
      inst ext of Ext
