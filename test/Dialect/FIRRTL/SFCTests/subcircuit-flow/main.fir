; Main circuit ("remote")

circuit Top :
  module Bar :
    input a: UInt
    input b: UInt
    output o: UInt
    o <= add(a, not(b))

  module Foo :
    input clock : Clock
    input reset : Reset
    input dataIn : {a: {b: {c: UInt<42>}}, d: UInt<9001>, e: UInt<42>}
    output dataOut : {x: {y: {z: UInt<42>}}, w: UInt<9001>, u: UInt<42>}
    reg stuff : UInt<42>, clock
    reg stuff2 : UInt<9001>, clock

    stuff <= dataIn.a.b.c
    stuff2 <= dataIn.d
    dataOut.x.y.z <= stuff
    dataOut.w <= stuff2

    inst bar of Bar
    bar.a <= dataIn.a.b.c
    bar.b <= dataIn.e
    dataOut.u <= bar.o

  module Top :
    input clock : Clock
    input reset : UInt<1>
    input dataIn : {a: {b: {c: UInt<42>}}, d: UInt<9001>, e: UInt<42>}
    output zu: UInt

    inst foo of Foo
    foo.clock <= asClock(UInt<1>(0))
    foo.reset <= reset
    foo.dataIn <- dataIn
    zu <= and(foo.dataOut.x.y.z, foo.dataOut.u)
