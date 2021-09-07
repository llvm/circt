; RUN: true

circuit Top :
  module Foo :
    input clock : Clock
    input reset : Reset
    input dataIn : {a: {b: {c: UInt<42>}}, d: UInt<9001>}
    output dataOut : {x: {y: {z: UInt<42>}}, w: UInt<9001>}

    reg stuff : UInt<42>
    reg stuff2 : UInt<9001>
    stuff <= dataIn.a.b.c
    stuff2 <= dataIn.d
    dataOut.x.y.z <= stuff
    dataOut.w <= stuff2

  module Top :
    input clock : Clock
    input reset : UInt<1>

    inst foo of Foo
    foo.clock <= asClock(UInt<1>(0))
    foo.reset <= reset
    foo.dataIn is invalid
