; RUN: firtool %s
; RUN: firtool %s -preserve-aggregate=all -scalarize-top-module=false
circuit Forward :
  ; SPEC EXAMPLE BEGIN
  extmodule Foo :
    output p : Probe<UInt<3>[2]>[2]
    ; ...

  module Forward :
    output p : Probe<UInt>

    inst f of Foo
    define p = f.p[0][1]
  ; SPEC EXAMPLE END
