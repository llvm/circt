; RUN: firtool %s
; RUN: firtool %s -preserve-aggregate=all -scalarize-top-module=false
; NOT SPEC EXAMPLE
FIRRTL version 3.0.0

circuit MinimumWidth :
  module Uninferred :
    input x : UInt
    output r : Probe<UInt>
    define r = probe(x)

  module MinimumWidth :
    input x : UInt<4>
    output y : UInt

    inst u of Uninferred
    connect u.x, x
    connect y, read(u.r)

