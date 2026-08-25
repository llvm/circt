; RUN: firtool %s --format=fir --parse-only -verify-diagnostics --split-input-file --warn-on-implicit-truncation

; CHECK-LABEL: outOfOrderFields
FIRRTL version 4.0.0
circuit outOfOrderFields :
  public module outOfOrderFields :
    output a: { a: UInt<5>, b: UInt<3> }
    input b: { b: UInt<1>, a: UInt<6> }
    ; expected-error@+1 {{cannot connect non-equivalent type '!firrtl.bundle<b: uint<1>, a: uint<6>>' to '!firrtl.bundle<a: uint<5>, b: uint<3>>'}}
    connect a, b

; // -----

FIRRTL version 4.0.0
circuit TruncWarn :
  public module TruncWarn :
    input in : UInt<7>
    output out : UInt<4>
    ; expected-warning@below {{inserting implicit truncation}}
    connect out, in

