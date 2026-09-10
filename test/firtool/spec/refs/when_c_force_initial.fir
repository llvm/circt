; RUN: firtool %s
; RUN: firtool %s -preserve-aggregate=all -scalarize-public-modules=false
FIRRTL version 3.0.0

circuit WhenCForce :
  module RefMe :
    output p : RWProbe<UInt<4>>
    wire x : UInt<4>
    connect x, UInt(0)
    define p = rwprobe(x)

  module WhenCForce :
    input c : UInt<1>
    input x : UInt<4>


; SPEC EXAMPLE:
;   when c : force_initial(ref, x)
; END SPEC EXAMPLE
   inst r of RefMe
   when c : force_initial(r.p, x)
