; RUN: firtool %s -verify-diagnostics
; RUN: firtool %s -preserve-aggregate=all -scalarize-top-module=false -verify-diagnostics
FIRRTL version 3.0.0
circuit UTurnTH:
  module UTurnTH:
    input in : UInt<5>
    output out : UInt<5>

    inst rb of RefBouncing
    connect rb.x, in
    connect out, rb.y

  ; SPEC EXAMPLE BEGIN
  module UTurn:
    input in : Probe<UInt>
    output out : Probe<UInt>
    define out = in

  module RefBouncing:
    input x: UInt
    output y: UInt

    inst u1 of UTurn
    inst u2 of UTurn

    node n = x
    define u1.in = probe(n)
    define u2.in = u1.out

    connect y, read(u2.out) ; = x
  ; SPEC EXAMPLE END
