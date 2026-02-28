; RUN: firtool %s -split-input-file -verify-diagnostics

; Check that we resolve last connect semantics before detecting combinational
; loops.
FIRRTL version 4.0.0
circuit CombinationalLoop:
  public module CombinationalLoop:
    output o : UInt<1>
    wire w : UInt<1>
    connect w, w
    connect w, UInt<1>(0)
    connect o, w
