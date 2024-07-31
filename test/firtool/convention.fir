; RUN: firtool %s --format=fir --parse-only --scalarize-public-modules=false --scalarize-ext-modules=false --scalarize-internal-modules=false --preserve-aggregate=all | FileCheck --check-prefix=SCALARIZE_FFF %s
; RUN: firtool %s --format=fir --parse-only --scalarize-public-modules=true  --scalarize-ext-modules=false --scalarize-internal-modules=false --preserve-aggregate=all | FileCheck --check-prefix=SCALARIZE_TFF  %s
; RUN: firtool %s --format=fir --parse-only --scalarize-public-modules=false --scalarize-ext-modules=true  --scalarize-internal-modules=false --preserve-aggregate=all | FileCheck --check-prefix=SCALARIZE_FTF  %s
; RUN: firtool %s --format=fir --parse-only --scalarize-public-modules=true  --scalarize-ext-modules=true  --scalarize-internal-modules=false --preserve-aggregate=all | FileCheck --check-prefix=SCALARIZE_TTF %s

; RUN: firtool %s --format=fir --parse-only --scalarize-public-modules=false --scalarize-ext-modules=false --scalarize-internal-modules=true  --preserve-aggregate=all | FileCheck --check-prefix=SCALARIZE_FFT %s
; RUN: firtool %s --format=fir --parse-only --scalarize-public-modules=true  --scalarize-ext-modules=false --scalarize-internal-modules=true  --preserve-aggregate=all | FileCheck --check-prefix=SCALARIZE_TFT  %s
; RUN: firtool %s --format=fir --parse-only --scalarize-public-modules=false --scalarize-ext-modules=true  --scalarize-internal-modules=true  --preserve-aggregate=all | FileCheck --check-prefix=SCALARIZE_FTT  %s
; RUN: firtool %s --format=fir --parse-only --scalarize-public-modules=true  --scalarize-ext-modules=true  --scalarize-internal-modules=true  --preserve-aggregate=all | FileCheck --check-prefix=SCALARIZE_TTT %s


; Ensure that top module and ext modules are marked scalarized.

FIRRTL version 4.0.0
circuit Top :
  ; SCALARIZE_FFF-NOT: attributes {convention = #firrtl<convention scalarized>}
  ; SCALARIZE_TFF:     attributes {convention = #firrtl<convention scalarized>}
  ; SCALARIZE_FTF-NOT: attributes {convention = #firrtl<convention scalarized>}
  ; SCALARIZE_TTF:     attributes {convention = #firrtl<convention scalarized>}
  ; SCALARIZE_FFT-NOT: attributes {convention = #firrtl<convention scalarized>}
  ; SCALARIZE_TFT:     attributes {convention = #firrtl<convention scalarized>}
  ; SCALARIZE_FTT-NOT: attributes {convention = #firrtl<convention scalarized>}
  ; SCALARIZE_TTT:     attributes {convention = #firrtl<convention scalarized>}
  public module Top :
    output port: UInt<8>[2]
    port[0] <= UInt<8>(0)
    port[1] <= UInt<8>(0)

  ; SCALARIZE_FFF-NOT: attributes {convention = #firrtl<convention scalarized>}
  ; SCALARIZE_TFF-NOT: attributes {convention = #firrtl<convention scalarized>}
  ; SCALARIZE_FTF:     attributes {convention = #firrtl<convention scalarized>}
  ; SCALARIZE_TTF:     attributes {convention = #firrtl<convention scalarized>}
  ; SCALARIZE_FFT-NOT: attributes {convention = #firrtl<convention scalarized>}
  ; SCALARIZE_TFT-NOT: attributes {convention = #firrtl<convention scalarized>}
  ; SCALARIZE_FTT:     attributes {convention = #firrtl<convention scalarized>}
  ; SCALARIZE_TTT:     attributes {convention = #firrtl<convention scalarized>}
  extmodule External :
    output port: UInt<8>[2]

  ; SCALARIZE_FFF-NOT: attributes {convention = #firrtl<convention scalarized>}
  ; SCALARIZE_TFF-NOT: attributes {convention = #firrtl<convention scalarized>}
  ; SCALARIZE_FTF-NOT: attributes {convention = #firrtl<convention scalarized>}
  ; SCALARIZE_TTF-NOT: attributes {convention = #firrtl<convention scalarized>}
  ; SCALARIZE_FFT:     attributes {convention = #firrtl<convention scalarized>}
  ; SCALARIZE_TFT:     attributes {convention = #firrtl<convention scalarized>}
  ; SCALARIZE_FTT:     attributes {convention = #firrtl<convention scalarized>}
  ; SCALARIZE_TTT:     attributes {convention = #firrtl<convention scalarized>}
  module Internal :
    output port: UInt<8>[2]
    port[0] <= UInt<8>(0)
    port[1] <= UInt<8>(0)
