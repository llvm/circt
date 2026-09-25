; REQUIRES: verilator
; RUN: firtool --probes-to-signals -verilog %s > %t.sv
; RUN: circt-rtl-sim.py %t.sv --cycles 16 2>&1 | FileCheck %s
;
; Test force/release synthesis for wires, registers, lowered aggregate fields,
; and probes.
; Test clocked force/release, release priority, reset interaction, and
; self-incrementing registers.

FIRRTL version 4.0.0
circuit top :
  module ClockedWire :
    input clock : Clock
    input en : UInt<1>
    input rel : UInt<1>
    input v : UInt<8>
    input d : UInt<8>
    output o : UInt<8>

    wire w : UInt<8>
    connect w, d
    connect o, w
    force(clock, en, rwprobe(w), v)
    release(clock, rel, rwprobe(w))

  ; Test force priority.
  module TwoForces :
    input clock : Clock
    input en1 : UInt<1>
    input en2 : UInt<1>
    input v1 : UInt<8>
    input v2 : UInt<8>
    input d : UInt<8>
    output o : UInt<8>

    wire w : UInt<8>
    connect w, d
    connect o, w
    force(clock, en1, rwprobe(w), v1)
    force(clock, en2, rwprobe(w), v2)

  ; Test force/release on a resettable register.
  module ForcedReg :
    input clock : Clock
    input rst : UInt<1>
    input en : UInt<1>
    input rel : UInt<1>
    input v : UInt<8>
    input d : UInt<8>
    output o : UInt<8>
    output p : RWProbe<UInt<8>>

    regreset r : UInt<8>, clock, rst, UInt<8>(0)
    connect r, d
    connect o, r
    define p = rwprobe(r)
    force(clock, en, rwprobe(r), v)
    release(clock, rel, rwprobe(r))

  module LocalForceChild :
    input clock : Clock
    input enL : UInt<1>
    input vL : UInt<8>
    input d : UInt<8>
    output o : UInt<8>
    output p : RWProbe<UInt<8>>

    wire w : UInt<8>
    connect w, d
    connect o, w
    define p = rwprobe(w)
    force(clock, enL, rwprobe(w), vL)

  module BundleField :
    input clock : Clock
    input en : UInt<1>
    input v : UInt<8>
    input x : UInt<8>
    input y : UInt<8>
    output oa : UInt<8>
    output ob : UInt<8>

    wire w : {a : UInt<8>, b : UInt<8>}
    connect w.a, x
    connect w.b, y
    connect oa, w.a
    connect ob, w.b
    force(clock, en, rwprobe(w.a), v)

  ; Test force/release on a self-incrementing register.
  module SelfIncReg :
    input clock : Clock
    input rst : UInt<1>
    input en : UInt<1>
    input rel : UInt<1>
    input v : UInt<8>
    output o : UInt<8>
    output p : RWProbe<UInt<8>>

    regreset r : UInt<8>, clock, rst, UInt<8>(0)
    connect r, add(r, UInt<8>(1))
    connect o, r
    define p = rwprobe(r)
    force(clock, en, rwprobe(r), v)
    release(clock, rel, rwprobe(r))

  public module top :
    input clk : Clock
    input rst : UInt<1>

    regreset cycle : UInt<8>, clk, rst, UInt<8>(0)
    connect cycle, add(cycle, UInt<8>(1))
    node nreset = eq(rst, UInt<1>(0))
    ; Test force/release with a non-constant driver.
    node d = pad(rst, 8)
    node y = pad(nreset, 8)
    ; Test a changing force value.
    node vLive = mux(gt(cycle, UInt<8>(2)), UInt<8>(0hBB), UInt<8>(0hAA))

    inst clockedWire of ClockedWire
    connect clockedWire.clock, clk
    connect clockedWire.en, eq(cycle, UInt<8>(1))
    connect clockedWire.rel, eq(cycle, UInt<8>(7))
    connect clockedWire.v, vLive
    connect clockedWire.d, d

    ; Test simultaneous force and release.
    inst concurrentForce of ClockedWire
    connect concurrentForce.clock, clk
    connect concurrentForce.en, eq(cycle, UInt<8>(1))
    connect concurrentForce.rel, eq(cycle, UInt<8>(1))
    connect concurrentForce.v, UInt<8>(0hAA)
    connect concurrentForce.d, d

    inst priorityForces of TwoForces
    connect priorityForces.clock, clk
    connect priorityForces.en1, or(eq(cycle, UInt<8>(1)), eq(cycle, UInt<8>(5)))
    connect priorityForces.en2, or(eq(cycle, UInt<8>(3)), eq(cycle, UInt<8>(5)))
    connect priorityForces.v1, UInt<8>(0hA1)
    connect priorityForces.v2, UInt<8>(0hA2)
    connect priorityForces.d, d

    inst resettableReg of ForcedReg
    connect resettableReg.clock, clk
    connect resettableReg.rst, or(rst, eq(cycle, UInt<8>(3)))
    connect resettableReg.en, eq(cycle, UInt<8>(1))
    connect resettableReg.rel, eq(cycle, UInt<8>(7))
    connect resettableReg.v, UInt<8>(0hAA)
    connect resettableReg.d, d

    inst parentChildForce of LocalForceChild
    connect parentChildForce.clock, clk
    connect parentChildForce.enL, or(eq(cycle, UInt<8>(1)), eq(cycle, UInt<8>(5)))
    connect parentChildForce.vL, mux(gt(cycle, UInt<8>(4)), UInt<8>(0hC1), UInt<8>(0hA1))
    connect parentChildForce.d, d
    force(clk, eq(cycle, UInt<8>(3)), parentChildForce.p, UInt<8>(0hA2))
    release(clk, eq(cycle, UInt<8>(7)), parentChildForce.p)

    inst bundleField of BundleField
    connect bundleField.clock, clk
    connect bundleField.en, eq(cycle, UInt<8>(1))
    connect bundleField.v, vLive
    connect bundleField.x, d
    connect bundleField.y, y

    inst selfIncrementingReg of SelfIncReg
    connect selfIncrementingReg.clock, clk
    connect selfIncrementingReg.rst, rst
    connect selfIncrementingReg.en, eq(cycle, UInt<8>(1))
    connect selfIncrementingReg.rel, eq(cycle, UInt<8>(7))
    connect selfIncrementingReg.v, UInt<8>(0h10)

    ; CHECK: idle clockedWire={{0*}}00 concurrentForce={{0*}}00 priorityForces={{0*}}00 resettableReg={{0*}}00 parentChildForce={{0*}}00 bundleA={{0*}}00 bundleB={{0*}}01 selfIncrementingReg={{0*}}00
    printf(clk, and(nreset, eq(cycle, UInt<8>(0))), "idle clockedWire=%x concurrentForce=%x priorityForces=%x resettableReg=%x parentChildForce=%x bundleA=%x bundleB=%x selfIncrementingReg=%x\n", clockedWire.o, concurrentForce.o, priorityForces.o, resettableReg.o, parentChildForce.o, bundleField.oa, bundleField.ob, selfIncrementingReg.o)
    ; CHECK: forced clockedWire={{0*}}aa concurrentForce={{0*}}00 priorityForces={{0*}}a1 resettableReg={{0*}}aa parentChildForce={{0*}}a1 bundleA={{0*}}aa bundleB={{0*}}01 selfIncrementingReg={{0*}}10
    printf(clk, and(nreset, eq(cycle, UInt<8>(2))), "forced clockedWire=%x concurrentForce=%x priorityForces=%x resettableReg=%x parentChildForce=%x bundleA=%x bundleB=%x selfIncrementingReg=%x\n", clockedWire.o, concurrentForce.o, priorityForces.o, resettableReg.o, parentChildForce.o, bundleField.oa, bundleField.ob, selfIncrementingReg.o)
    ; CHECK: live clockedWire={{0*}}bb resettableReg={{0*}}aa bundleA={{0*}}bb bundleB={{0*}}01
    printf(clk, and(nreset, eq(cycle, UInt<8>(3))), "live clockedWire=%x resettableReg=%x bundleA=%x bundleB=%x\n", clockedWire.o, resettableReg.o, bundleField.oa, bundleField.ob)
    ; CHECK: priority priorityForces={{0*}}a2 parentChildForce={{0*}}a2
    printf(clk, and(nreset, eq(cycle, UInt<8>(4))), "priority priorityForces=%x parentChildForce=%x\n", priorityForces.o, parentChildForce.o)
    ; CHECK: priority2 priorityForces={{0*}}a2 parentChildForce={{0*}}c1
    printf(clk, and(nreset, eq(cycle, UInt<8>(6))), "priority2 priorityForces=%x parentChildForce=%x\n", priorityForces.o, parentChildForce.o)
    ; CHECK: released clockedWire={{0*}}00 resettableReg={{0*}}00 parentChildForce={{0*}}00 selfIncrementingReg={{0*}}11
    printf(clk, and(nreset, eq(cycle, UInt<8>(8))), "released clockedWire=%x resettableReg=%x parentChildForce=%x selfIncrementingReg=%x\n", clockedWire.o, resettableReg.o, parentChildForce.o, selfIncrementingReg.o)
    ; CHECK: PASS
    printf(clk, and(nreset, eq(cycle, UInt<8>(10))), "PASS\n")
    stop(clk, and(nreset, eq(cycle, UInt<8>(10))), 0)
