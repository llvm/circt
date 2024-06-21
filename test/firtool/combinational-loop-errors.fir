; RUN: firtool %s -disable-layers=A -split-input-file -verify-diagnostics

; Check that we detect combinational loops in specialized layers.
FIRRTL version 4.0.0
circuit CombinationalLoop:
  layer A, bind:
  ; expected-error @below {{detected combinational cycle in a FIRRTL module, sample path: CombinationalLoop.{w <- w}}}
  public module CombinationalLoop:
    layerblock A:
      wire w : UInt<1>
      connect w, w
