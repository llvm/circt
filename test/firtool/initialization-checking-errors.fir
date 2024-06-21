; RUN: firtool %s -disable-layers=A -split-input-file -verify-diagnostics

; Check that we detect uninitialized values in specialized layers.
FIRRTL version 4.0.0
circuit Uninitialized:
  layer A, bind:
  public module Uninitialized:
    layerblock A:
      ; expected-error @below {{sink "w" not fully initialized in "Uninitialized"}}
      wire w : UInt<1>
