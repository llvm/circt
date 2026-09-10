; RUN: rm -rf %t && mkdir %t && firtool --split-verilog --annotation-file %s.anno.json --annotation-file %s.extract.anno.json -o %t %s && cd %t && verilator --sv --lint-only Top.sv bindings.sv
; RUN: firtool --verilog --annotation-file %s.anno.json %s > %t.one-file.sv && verilator --sv --lint-only %t.one-file.sv
; REQUIRES: verilator

FIRRTL version 4.0.0
circuit Top :
  module Submodule :
    input clock : Clock
    input reset : Reset
    input in : { uint : UInt<1>, vec : UInt<1>[2], vecOfBundle : { uint : UInt<4>, sint : SInt<2>}[2], otherOther : { other : { uint : UInt<4>, sint : SInt<2>}}}
    output out : { uint : UInt<1>, vec : UInt<1>[2], vecOfBundle : { uint : UInt<4>, sint : SInt<2>}[2], otherOther : { other : { uint : UInt<4>, sint : SInt<2>}}}

    wire w : { uint : UInt<1>, vec : UInt<1>[2], vecOfBundle : { uint : UInt<4>, sint : SInt<2>}[2], otherOther : { other : { uint : UInt<4>, sint : SInt<2>}}}
    connect w.otherOther.other.sint, in.otherOther.other.sint
    connect w.otherOther.other.uint, in.otherOther.other.uint
    connect w.vecOfBundle[0].sint, in.vecOfBundle[0].sint
    connect w.vecOfBundle[0].uint, in.vecOfBundle[0].uint
    connect w.vecOfBundle[1].sint, in.vecOfBundle[1].sint
    connect w.vecOfBundle[1].uint, in.vecOfBundle[1].uint
    connect w.vec[0], in.vec[0]
    connect w.vec[1], in.vec[1]
    connect w.uint, in.uint
    connect out.otherOther.other.sint, w.otherOther.other.sint
    connect out.otherOther.other.uint, w.otherOther.other.uint
    connect out.vecOfBundle[0].sint, w.vecOfBundle[0].sint
    connect out.vecOfBundle[0].uint, w.vecOfBundle[0].uint
    connect out.vecOfBundle[1].sint, w.vecOfBundle[1].sint
    connect out.vecOfBundle[1].uint, w.vecOfBundle[1].uint
    connect out.vec[0], w.vec[0]
    connect out.vec[1], w.vec[1]
    connect out.uint, w.uint

  module MyView_companion :
    output io : { }

    wire _WIRE : UInt<1>
    connect _WIRE, UInt<1>(0h0)

  module DUT :
    input clock : Clock
    input reset : Reset
    input in : { uint : UInt<1>, vec : UInt<1>[2], vecOfBundle : { uint : UInt<4>, sint : SInt<2>}[2], otherOther : { other : { uint : UInt<4>, sint : SInt<2>}}}
    output out : { uint : UInt<1>, vec : UInt<1>[2], vecOfBundle : { uint : UInt<4>, sint : SInt<2>}[2], otherOther : { other : { uint : UInt<4>, sint : SInt<2>}}}

    wire w : { uint : UInt<1>, vec : UInt<1>[2], vecOfBundle : { uint : UInt<4>, sint : SInt<2>}[2], otherOther : { other : { uint : UInt<4>, sint : SInt<2>}}}
    inst submodule of Submodule
    connect submodule.clock, clock
    connect submodule.reset, reset
    connect w.otherOther.other.sint, in.otherOther.other.sint
    connect w.otherOther.other.uint, in.otherOther.other.uint
    connect w.vecOfBundle[0].sint, in.vecOfBundle[0].sint
    connect w.vecOfBundle[0].uint, in.vecOfBundle[0].uint
    connect w.vecOfBundle[1].sint, in.vecOfBundle[1].sint
    connect w.vecOfBundle[1].uint, in.vecOfBundle[1].uint
    connect w.vec[0], in.vec[0]
    connect w.vec[1], in.vec[1]
    connect w.uint, in.uint
    connect submodule.in.otherOther.other.sint, w.otherOther.other.sint
    connect submodule.in.otherOther.other.uint, w.otherOther.other.uint
    connect submodule.in.vecOfBundle[0].sint, w.vecOfBundle[0].sint
    connect submodule.in.vecOfBundle[0].uint, w.vecOfBundle[0].uint
    connect submodule.in.vecOfBundle[1].sint, w.vecOfBundle[1].sint
    connect submodule.in.vecOfBundle[1].uint, w.vecOfBundle[1].uint
    connect submodule.in.vec[0], w.vec[0]
    connect submodule.in.vec[1], w.vec[1]
    connect submodule.in.uint, w.uint
    connect out.otherOther.other.sint, submodule.out.otherOther.other.sint
    connect out.otherOther.other.uint, submodule.out.otherOther.other.uint
    connect out.vecOfBundle[0].sint, submodule.out.vecOfBundle[0].sint
    connect out.vecOfBundle[0].uint, submodule.out.vecOfBundle[0].uint
    connect out.vecOfBundle[1].sint, submodule.out.vecOfBundle[1].sint
    connect out.vecOfBundle[1].uint, submodule.out.vecOfBundle[1].uint
    connect out.vec[0], submodule.out.vec[0]
    connect out.vec[1], submodule.out.vec[1]
    connect out.uint, submodule.out.uint
    inst MyView_companion of MyView_companion

  public module Top :
    input clock : Clock
    input reset : UInt<1>
    input in : { uint : UInt<1>, vec : UInt<1>[2], vecOfBundle : { uint : UInt<4>, sint : SInt<2>}[2], otherOther : { other : { uint : UInt<4>, sint : SInt<2>}}}
    output out : { uint : UInt<1>, vec : UInt<1>[2], vecOfBundle : { uint : UInt<4>, sint : SInt<2>}[2], otherOther : { other : { uint : UInt<4>, sint : SInt<2>}}}

    inst dut of DUT
    connect dut.clock, clock
    connect dut.reset, reset
    connect dut.in.otherOther.other.sint, in.otherOther.other.sint
    connect dut.in.otherOther.other.uint, in.otherOther.other.uint
    connect dut.in.vecOfBundle[0].sint, in.vecOfBundle[0].sint
    connect dut.in.vecOfBundle[0].uint, in.vecOfBundle[0].uint
    connect dut.in.vecOfBundle[1].sint, in.vecOfBundle[1].sint
    connect dut.in.vecOfBundle[1].uint, in.vecOfBundle[1].uint
    connect dut.in.vec[0], in.vec[0]
    connect dut.in.vec[1], in.vec[1]
    connect dut.in.uint, in.uint
    connect out.otherOther.other.sint, dut.out.otherOther.other.sint
    connect out.otherOther.other.uint, dut.out.otherOther.other.uint
    connect out.vecOfBundle[0].sint, dut.out.vecOfBundle[0].sint
    connect out.vecOfBundle[0].uint, dut.out.vecOfBundle[0].uint
    connect out.vecOfBundle[1].sint, dut.out.vecOfBundle[1].sint
    connect out.vecOfBundle[1].uint, dut.out.vecOfBundle[1].uint
    connect out.vec[0], dut.out.vec[0]
    connect out.vec[1], dut.out.vec[1]
    connect out.uint, dut.out.uint
