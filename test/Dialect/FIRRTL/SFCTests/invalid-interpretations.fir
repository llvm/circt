; RUN: firtool -split-input-file -verilog %s | FileCheck %s

; This test checks end-to-end compliance with the Scala FIRRTL Compiler (SFC)
; context-sensitive interpretation of invalid.

; CHECK-LABEL: module InvalidInterpretations
circuit InvalidInterpretations:
  module InvalidInterpretations:
    input clock: Clock
    input reset: UInt<1>
    input cond: UInt<1>
    input a: UInt<8>
    output out_when: UInt<8>
    output out_validif: UInt<8>
    output out_reg: UInt<8>
    output out_mux: UInt<8>
    output out_add: UInt<9>

    wire inv: UInt<8>
    inv is invalid

    reg r: UInt<8>, clock with : (reset => (reset, inv))
    r <= a
    out_reg <= r
    ; Interpretation 1: Invalid is undefined if used as the initialization value
    ; of a register in a module-scoped analysis that looks through connects.
    ; CHECK:       always @(posedge clock)
    ; CHECK-NOT:     if (reset)

    out_when is invalid
    when cond:
      out_when <= a
    ; Interpretation 2: Invalid is undefined when used as a default value.
    ; CHECK:       assign out_when = a;

    out_validif <= validif(cond, a)
    ; Interpretation 3: Invalid is undefined as the false leg of a validif.
    ; CHECK:       assign out_validif = a;

    out_mux <= mux(cond, a, inv)
    out_add <= add(a, inv)
    ; Interpretation 4: Invalid is zero otherwise.
    ; CHECK:       assign out_mux = cond ? a : 8'h0;
    ; CHECK-NEXT:  assign out_add = {1'h0, a};
