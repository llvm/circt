; RUN: firtool %s --format=fir --parse-only -verify-diagnostics

; CHECK-LABEL: outOfOrderFields
FIRRTL version 4.0.0
circuit outOfOrderFields :
  public module outOfOrderFields :
    output a: { a: UInt<5>, b: UInt<3> }
    input b: { b: UInt<1>, a: UInt<6> }
    ; expected-error@+1 {{cannot connect non-equivalent type '!firrtl.bundle<b: uint<1>, a: uint<6>>' to '!firrtl.bundle<a: uint<5>, b: uint<3>>'}}
    a <= b
