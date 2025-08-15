; RUN: firtool %s | FileCheck %s

FIRRTL version 5.0.0
circuit Top:
  layer Verification, bind, "verification":

  public module Top:
    input  c : Clock
    input  i : UInt<1>
    input  j : UInt<1>
    input  d : UInt<1>
    layerblock Verification:
      cmem memory : UInt<1>[2]
      infer mport rd = memory[i], c
      infer mport wr = memory[j], c
      connect wr, d
      assert(c, rd, UInt<1>(1), "hello")

; CHECK: ----- 8< ----- FILE "verification/memory_2x1.sv" ----- 8< -----
