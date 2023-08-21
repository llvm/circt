; RUN: firtool %s -verify-diagnostics
; RUN: firtool %s -preserve-aggregate=all -scalarize-top-module=false -verify-diagnostics
circuit UnusedInputRef :
  ; expected-error @+3 {{input probes not yet supported}}
  ; SPEC EXAMPLE BEGIN
  module UnusedInputRef :
    input r : Probe<UInt<1>>

  module ForwardDownwards :
    input in : UInt<1>

    inst u of UnusedInputRef
    define u.r = probe(in)
  ; SPEC EXAMPLE END
