; RUN: firtool --parse-only %s | FileCheck %s --check-prefixes=COMMON,DEFAULT
; RUN: firtool --parse-only %s --default-layer-specialization=none | FileCheck %s --check-prefixes=COMMON,NONE
; RUN: firtool --parse-only %s --default-layer-specialization=enable | FileCheck %s --check-prefixes=COMMON,ENABLE
; RUN: firtool --parse-only %s --default-layer-specialization=disable | FileCheck %s --check-prefixes=COMMON,DISABLE

; Check that default layer specialization attaches a circuit attribute.

FIRRTL version 4.0.0
; COMMON:  firrtl.circuit "LayerSpecialization"
; DEFAULT-NOT:  default_layer_specialization
; NONE-NOT:     default_layer_specialization
; ENABLE-SAME:  default_layer_specialization = #firrtl<layerspecialization enable>
; DISABLE-SAME: default_layer_specialization = #firrtl<layerspecialization disable>
circuit LayerSpecialization:
  extmodule LayerSpecialization:
