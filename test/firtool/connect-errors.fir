; RUN: firtool %s --format=fir --parse-only --split-input-file -verify-diagnostics


; Truncation is an error.
FIRRTL version 4.0.0
circuit truncatingIntegerConnect :
 public module truncatingIntegerConnect :
   output a: UInt<3>
   input b: UInt<5>
   ; expected-error @+1 {{destination '!firrtl.uint<3>' is not as wide as the source '!firrtl.uint<5>'}}
   connect a, b

; // -----

FIRRTL version 4.0.0
circuit truncatingBundleConnect :
 public module truncatingBundleConnect :
   output a: { a: UInt<5>, b: UInt<3> }
   input b: { a: UInt<6>, b: UInt<1> }
   ; expected-error @+1 {{destination '!firrtl.bundle<a: uint<5>, b: uint<3>>' is not as wide as the source '!firrtl.bundle<a: uint<6>, b: uint<1>>'}}
   connect a, b

; // -----

; CHECK-LABEL: outOfOrderFields
FIRRTL version 4.0.0
circuit outOfOrderFields :
  public module outOfOrderFields :
    output a: { a: UInt<5>, b: UInt<3> }
    input b: { b: UInt<1>, a: UInt<6> }
    ; expected-error @+1 {{cannot connect non-equivalent type '!firrtl.bundle<b: uint<1>, a: uint<6>>' to '!firrtl.bundle<a: uint<5>, b: uint<3>>'}}
    connect a, b

