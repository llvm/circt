; RUN: firtool %s
; RUN: firtool %s -preserve-aggregate=all -scalarize-top-module=false
FIRRTL version 3.0.0

circuit Top :
  ; SPEC EXAMPLE BEGIN
  module Top:
    input x : {a: UInt<2>, flip b: UInt<2>}
    output y : {a: UInt<2>, flip b: UInt<2>}

    inst d of DUT
    connect d.x, x
    connect y, d.y

    wire val : {a: UInt<2>, b: UInt<2>}
    connect val.a, UInt<2>(1)
    connect val.b, UInt<2>(2)

    ; Force takes a RWProbe and overrides the target with 'val'.
    force_initial(d.xp, val)

  module DUT :
    input x : {a: UInt<2>, flip b: UInt<2>}
    output y : {a: UInt<2>, flip b: UInt<2>}
    output xp : RWProbe<{a: UInt<2>, b: UInt<2>}>

    ; Force drives p.a, p.b, y.a, and x.b, but not y.b and x.a
    wire p : {a: UInt<2>, flip b: UInt<2>}
    define xp = rwprobe(p)
    connect p, x
    connect y, p
  ; SPEC EXAMPLE END
