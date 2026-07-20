; RUN: circt-translate -import-firrtl -verify-diagnostics %s | circt-opt | FileCheck %s

; Importing a register whose reset operand is `AsyncReset`-typed must stamp the
; authoritative `resetType = AsyncReset` (1 : i32) attribute so async-vs-sync is
; recorded in the IR rather than inferred from the operand type downstream.
; A synchronous (UInt<1>) reset keeps the elided default and carries no
; `resetType` attribute.
FIRRTL version 4.0.0
circuit Migration :
  public module Migration :
    input clock : Clock
    input areset : AsyncReset
    input sreset : UInt<1>
    input in : UInt<8>

    ; CHECK: %ar = firrtl.regreset interesting_name %clock, %areset, %in {resetType = 1 : i32} : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<8>, !firrtl.uint<8>
    regreset ar : UInt<8>, clock, areset, in

    ; CHECK: %sr = firrtl.regreset interesting_name %clock, %sreset, %in : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>
    ; CHECK-NOT: resetType
    regreset sr : UInt<8>, clock, sreset, in
