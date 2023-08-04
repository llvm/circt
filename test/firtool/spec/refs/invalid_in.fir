; RUN: firtool %s --verify-diagnostics
; RUN: firtool %s -preserve-aggregate=all -scalarize-top-module=false --verify-diagnostics
FIRRTL version 3.0.0
circuit FooUser:
  ; Invalid, per spec
  ; expected-error @+3 {{input probes not yet supported}}
  ; SPEC EXAMPLE BEGIN
  module Foo:
     input in : Probe<UInt>
     output out : UInt

     connect out, read(in)
  ; SPEC EXAMPLE END
  module FooUser:
    output out : UInt

    node n = UInt<2>(1)
    inst f of Foo
    connect f.in, probe(n)
    connect out, f.out
