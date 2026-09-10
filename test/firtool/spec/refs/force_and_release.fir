; RUN: firtool %s
; RUN: firtool %s -preserve-aggregate=all -scalarize-public-modules=false
FIRRTL version 3.0.0
circuit ForceAndRelease:
  ; SPEC EXAMPLE BEGIN
  module ForceAndRelease:
    input a: UInt<2>
    input clock : Clock
    input cond : UInt<1>
    output o : UInt<3>

    inst r of AddRefs
    connect o, r.sum

    force(clock, cond, r.a, a)
    release(clock, not(cond), r.a)

  ; SPEC EXAMPLE END
  ; include "force_addrefs.fir"
  module AddRefs:
    output a : RWProbe<UInt<2>>
    output b : RWProbe<UInt<2>>
    output c : RWProbe<UInt<2>>
    output sum : UInt<3>

    ; XXX: modified, workaround inability create non-const node w/literal initializer.
    wire w : UInt<2>
    connect w, UInt<2>(0)
    node x = w
    node y = w
    node z = w
    connect sum, add(x, add(y, z))

    define a = rwprobe(x)
    define b = rwprobe(y)
    define c = rwprobe(z)
