; Subcircuit ("local")

circuit Sub :
  extmodule SubExtern :
    input clockIn : Clock
    output clockOut : Clock
    input someInput : { u: UInt<42>, v: UInt<9001>, w: UInt<1>[2] }
    output someOutput : { u: UInt<42>, v: UInt<9001>, w: UInt<1>[2] }

  module Sub :
    wire clock_source : Clock
    wire clock_sink : Clock
    wire data_source : { u: UInt<42>, v: UInt<9001>, w: UInt<1>[2] }
    wire data_sink : { u: UInt<42>, v: UInt<9001>, w: UInt<1>[2] }
    clock_source is invalid
    data_source is invalid

    inst ext of SubExtern
    ext.clockIn <= clock_source
    ext.someInput <= data_source
    clock_sink <= ext.clockOut
    data_sink <= ext.someOutput
