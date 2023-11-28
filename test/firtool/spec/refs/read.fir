; RUN: firtool %s
; RUN: firtool %s -preserve-aggregate=all -scalarize-public-modules=false
FIRRTL version 3.0.0

circuit Foo :
  ; SPEC EXAMPLE BEGIN
  module Foo :
    output p : Probe<UInt>
    ; ...
    wire x : UInt<4> ; XXX: ADDED
    invalidate x ; XXX: ADDED
    define p = probe(x) ; XXX: ADDED

  module Bar :
    output x : UInt

    inst f of Foo
    connect x, read(f.p) ; indirectly access the probed data
  ; SPEC EXAMPLE END
