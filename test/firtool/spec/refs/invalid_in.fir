; RUN: firtool %s --verify-diagnostics
; RUN: firtool %s -preserve-aggregate=all -scalarize-top-module=false --verify-diagnostics
FIRRTL version 3.0.0
circuit FooUser:
  ; Invalid, per spec
  ; expected-error @+7 {{input probes cannot be used}}
  ; expected-note @+3 {{input probe here}}
  ; SPEC EXAMPLE BEGIN
  module Foo:
     input in : Probe<UInt>
     output out : UInt

     connect out, read(in)
  ; SPEC EXAMPLE END
  module FooUser:
    input in : UInt<2>
    output out : UInt

    inst f of Foo
    define f.in = probe(in)
    connect out, f.out
