; RUN: firtool %s | FileCheck %s

; CHECK:      wire _GEN = 1'h0;
; CHECK:      if (reset)
; CHECK-NEXT:   r1 <= 1'h0;
; CHECK:      if (reset)
; CHECK-NEXT:   r1 = 1'h0;
; CHECK:      `define ref_Top_probe _GEN

FIRRTL version 6.0.0
circuit Top:
  public module Top:
    input clock: Clock
    input reset: AsyncReset
    input d: UInt<1>
    output q: UInt<1>
    output probe: Probe<UInt<1>>

    node v = UInt<1>(0)
    define probe = probe(v)

    regreset r1: UInt<1>, clock, reset, v
    connect r1, d
    connect q, r1
