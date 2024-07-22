; RUN: firtool %s -disable-layers=X,Y
; RUN: firtool %s -enable-layers=X,Y

; https://github.com/llvm/circt/issues/7345
; Check that we can specify more than one layer in the command line options.

FIRRTL version 4.0.0
circuit Foo:
  layer X, bind:
  layer Y, bind:
  public module Foo:
