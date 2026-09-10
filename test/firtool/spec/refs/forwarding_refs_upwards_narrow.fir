; RUN: firtool %s
; RUN: firtool %s -preserve-aggregate=all -scalarize-public-modules=false
FIRRTL version 4.0.0
circuit Forward :
  ; SPEC EXAMPLE BEGIN
  extmodule Foo :
    output p : Probe<UInt<3>[2]>[2]
    ; ...

  public module Forward :
    output p : Probe<UInt<3>>

    inst f of Foo
    define p = f.p[0][1]
  ; SPEC EXAMPLE END
