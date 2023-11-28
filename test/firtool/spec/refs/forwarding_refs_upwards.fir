; RUN: firtool %s
; RUN: firtool %s -preserve-aggregate=all -scalarize-public-modules=false
circuit Forward:
  ; SPEC EXAMPLE BEGIN
  extmodule Foo : ; XXX: module -> extmodule
    output p : Probe<UInt<3>> ; XXX: added width
    ; ...

  module Forward :
    output p : Probe<UInt>

    inst f of Foo
    define p = f.p
  ; SPEC EXAMPLE END
