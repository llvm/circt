; RUN: firtool --ir-fir %s | FileCheck %s --check-prefix=ENABLED
; RUN: firtool --ir-fir --disable-object-promotion %s | FileCheck %s --check-prefix=DISABLED
; RUN: firtool --ir-hw %s | FileCheck %s --check-prefix=IR-HW
; RUN: firtool --ir-hw --disable-object-promotion %s | FileCheck %s --check-prefix=IR-HW

; IR-HW-LABEL: om.class @ObjectPromotion_Class
; IR-HW: om.object @Foo(%basepath, %in)

FIRRTL version 6.0.0

circuit ObjectPromotion:
  public module ObjectPromotion:
    input in : String
    output out : AnyRef
    ; ENABLED-LABEL: firrtl.module @ObjectPromotion
    ; ENABLED: %obj = firrtl.object.create(%in) : (!firrtl.string) -> !firrtl.class<@Foo(in value: !firrtl.string)>
    ; DISABLED-LABEL: firrtl.module @ObjectPromotion
    ; DISABLED: %obj = firrtl.object @Foo(in value: !firrtl.string)
    object obj of Foo
    propassign obj.value, in
    propassign out, obj

  extclass Foo:
    input value : String
