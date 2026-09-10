; RUN: firtool %s
; RUN: firtool %s -preserve-aggregate=all -scalarize-public-modules=false
FIRRTL version 3.0.0
circuit Refs:
  ; SPEC EXAMPLE BEGIN
  module Refs:
    input clock:  Clock
    input p : {x: UInt<1>, flip y : UInt<3>} ; XXX: modified, for init
    output a : Probe<{x: UInt, y: UInt}> ; read-only ref. to wire 'p'
    output b : RWProbe<UInt> ; force-able ref. to node 'q', inferred width.
    output c : Probe<UInt> ; read-only ref. to register 'r', inferred width. ; XXX: modified, needs width
    output d : Probe<Clock> ; ref. to input clock port

    connect p.y, UInt<3>(0) ; XXX: modified, for init
    define a = probe(p) ; probe is passive
    node q = p.x ; XXX: modified, workaround inability create non-const node w/literal initializer.
    define b = rwprobe(q)
    reg r: UInt, clock
    connect r, p.x ; XXX: modified, initialize register
    define c = probe(r)
    define d = probe(clock)
  ; SPEC EXAMPLE END
