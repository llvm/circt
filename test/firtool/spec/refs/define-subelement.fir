; RUN: firtool %s
; RUN: firtool %s -preserve-aggregate=all -scalarize-top-module=false
FIRRTL version 3.0.0
circuit Foo:
  ; SPEC EXAMPLE BEGIN
  module Foo:
    input x : UInt<3> ; XXX: width added
    output y : {x: UInt, p: Probe<UInt>}
    output z : Probe<UInt>[2]

    wire w : UInt
    connect w, x
    connect y.x, w

    define y.p = probe(w)
    define z[0] = probe(w)
    define z[1] = probe(w)
   ; SPEC EXAMPLE END
