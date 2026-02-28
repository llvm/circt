; RUN: firtool %s --inline-input-only-modules | FileCheck %s --check-prefix INLINE
; RUN: firtool %s | FileCheck %s


FIRRTL version 5.0.0
circuit Top:
  ; INLINE-NOT: module InputOnly
  ; INLINE: `define ref_Top_probe input_probe
  ; CHECK: module InputOnly
  ; CHECK:  `define ref_Top_probe input_only.input_probe
  module InputOnly:
    ; Non-hardware ports must not be counted as output
    input input: UInt<1>
    output probe: Probe<UInt<1>>
    output x: Integer

    define probe = probe(input)
    propassign x, Integer(1)

  public module Top:
    input input: UInt<1>
    output probe: Probe<UInt<1>>
    output x: Integer

    inst input_only of InputOnly
    connect input_only.input, input
    define probe = input_only.probe
    propassign x, input_only.x
