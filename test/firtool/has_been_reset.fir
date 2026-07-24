; RUN: firtool --ir-hw %s | FileCheck %s
FIRRTL version 4.0.0

circuit Foo:
  public module Foo:
    input clock: Clock
    input reset1: Reset
    input reset2: Reset
    output hbr: UInt<1>[2]

    wire resetWire1 : Reset
    wire resetWire2 : Reset

    connect resetWire1, reset1
    connect resetWire2, reset2

    connect hbr[0], intrinsic(circt_has_been_reset : UInt<1>, clock, resetWire1)
    connect hbr[1], intrinsic(circt_has_been_reset : UInt<1>, clock, resetWire2)

; CHECK-LABEL: hw.module @Foo
; CHECK-NEXT: [[CLOCK:%.+]] = seq.from_clock %clock
; CHECK-NEXT: verif.has_been_reset [[CLOCK]], sync %reset1
; The '.fir' format cannot set the 'asyncReset' attribute, so a reset imported
; from '.fir' is synchronous; async 'has_been_reset' is covered in the MLIR test
; test/Conversion/FIRRTLToHW/intrinsics.mlir.
; CHECK-NEXT: verif.has_been_reset [[CLOCK]], sync %reset2
; CHECK-NEXT: hw.output
