; RUN: firtool %s
; RUN: firtool %s -preserve-aggregate=all -scalarize-top-module=false
FIRRTL version 3.0.0

circuit Foo :
  ; SPEC EXAMPLE BEGIN
  module Foo :
    output p : Probe<{a: UInt, b: UInt}>
    ; ...
    wire x : {a: UInt<5>, b: UInt<2>} ; XXX: ADDED
    invalidate x ; XXX: ADDED
    define p = probe(x) ; XXX: ADDED

  module Bar :
    output x : UInt

    inst f of Foo
    connect x, read(f.p.b) ; indirectly access the probed data
  ; SPEC EXAMPLE END

