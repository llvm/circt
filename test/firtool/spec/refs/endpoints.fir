; RUN: firtool %s
; RUN: firtool %s -preserve-aggregate=all -scalarize-public-modules=false
FIRRTL version 3.0.0
circuit Top : %[[
{ "class": "firrtl.transforms.DontTouchAnnotation", "target": "~Top|Top>producer_debug"},
{ "class": "firrtl.transforms.DontTouchAnnotation", "target": "~Top|Top>consumer_debug"}
]]
  ; SPEC EXAMPLE BEGIN
  module Consumer:
    input in : {a: UInt, pref: Probe<UInt>, flip cref: Probe<UInt>}
    ; ...
    node n = in.a
    define in.cref = probe(n)

  module Producer:
    output out : {a: UInt, pref: Probe<UInt>, flip cref: Probe<UInt>}
    wire x : UInt
    define out.pref = probe(x)
    ; ...
    connect out.a, x
    x <= UInt<3>(5) ; XXX: Modification: initialize x.

  module Connect:
    output out : {pref: Probe<UInt>, cref: Probe<UInt>}

    inst a of Consumer
    inst b of Producer

    ; Producer => Consumer
    connect a.in.a, b.out.a
    define a.in.pref = b.out.pref
    define b.out.cref = a.in.cref

    ; Send references out
    define out.pref = b.out.pref
    define out.cref = a.in.cref

  module Top:
    inst c of Connect

    node producer_debug = read(c.out.pref); ; Producer-side signal
    node consumer_debug = read(c.out.cref); ; Consumer-side signal
  ; SPEC EXAMPLE END
