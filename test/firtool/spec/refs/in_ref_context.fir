; RUN: firtool %s -verify-diagnostics
; RUN: firtool %s -preserve-aggregate=all -scalarize-top-module=false -verify-diagnostics
FIRRTL version 3.0.0
; expected-error @+27 {{input probes cannot be used}}
; expected-note @+19 {{input probe here}}
; SPEC EXAMPLE BEGIN
circuit Top:
  module Top:
    input in : UInt<4>
    output out : UInt

    inst ud1 of UpDown
    connect ud1.in, in
    define ud1.in_ref = ud1.r1

    inst ud2 of UpDown
    connect ud2.in, in
    define ud2.in_ref = ud2.r2

    connect out, add(ud1.out, ud2.out)

  module UpDown:
    input in : UInt<4>
    input in_ref : Probe<UInt<4>>
    output r1 : Probe<UInt<4>>
    output r2 : Probe<UInt<4>>
    output out : UInt

    ; In ud1, this is UpDown.n1, in ud2 this is UpDown.n2 .
    ; However, this is not supported as it cannot be both at once.
    connect out, read(in_ref)
    node n1 = and(in, UInt<4>(1))
    node n2 = and(in, UInt<4>(3))
    define r1 = probe(n1)
    define r2 = probe(n2)
; SPEC EXAMPLE END
