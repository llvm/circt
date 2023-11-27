; RUN: firtool %s -verify-diagnostics
; RUN: firtool %s -preserve-aggregate=all -scalarize-public-modules=false -verify-diagnostics
FIRRTL version 3.0.0
circuit ForwardDownwardsTH :
  module ForwardDownwardsTH :
    input in : UInt<1>
    output out : UInt<1>
    connect out, in ; Do something to quiet warnings about empty.

    inst fd of ForwardDownwards
    connect fd.in, in

  ; SPEC EXAMPLE BEGIN
  module UnusedInputRef :
    input r : Probe<UInt<1>>

  module ForwardDownwards :
    input in : UInt<1>

    inst u of UnusedInputRef
    define u.r = probe(in)
  ; SPEC EXAMPLE END
