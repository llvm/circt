circuit Top :
  module Foo :
    input clock : Clock
    input reset : Reset
    input dataIn : {a: {b: {c: UInt<42>}}}
    output dataOut : {x: {y: {z: UInt<42>}}}

    reg stuff : UInt<42>
    stuff <= dataIn.a.b.c
    dataOut.x.y.z <= dataOut

  module Top :
    input clock : Clock
    input reset : UInt<1>

    inst foo of Foo
    foo.clock <= asClock(UInt<1>(0))
    foo.reset <= reset
    foo.dataIn is invalid
