; RUN: firtool %s
; RUN: firtool %s -preserve-aggregate=all -scalarize-public-modules=false
FIRRTL version 3.0.0
circuit MyExternalModuleWithRefs :
  ; SPEC EXAMPLE BEGIN
  extmodule MyExternalModuleWithRefs :
    input foo : UInt<2>
    output mysignal : Probe<UInt<1>>
    output myreg : RWProbe<UInt<8>>
    ref mysignal is "a.b"
    ref myreg is "x.y"
  ; SPEC EXAMPLE END
