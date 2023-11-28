; RUN: firtool %s
; RUN: firtool %s -preserve-aggregate=all -scalarize-public-modules=false
FIRRTL version 3.0.0
circuit Foo :
  ; SPEC EXAMPLE BEGIN
  module Foo :
    input x : {a: UInt<3>, flip b: UInt} ; XXX: width on x.a
    output y : {a: UInt, flip b: UInt<3>} ; XXX: width on y.b
    output xp : Probe<{a: UInt, b: UInt}> ; passive

    wire p : {a: UInt, flip b: UInt} ; p is not passive
    define xp = probe(p)
    connect p, x
    connect y, p
   ; SPEC EXAMPLE END
